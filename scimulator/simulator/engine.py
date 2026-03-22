"""
Time-stepped drawdown simulation engine.

Phase 1 implementation: no active decisions. Pre-loaded inventory is consumed
by demand. Pre-scheduled inbound shipments arrive on their scheduled dates.
Fulfillment uses a simple lowest-variable-cost-first routing.

The engine is stateless — all state lives in DuckDB.
"""

import json
import logging
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional

import duckdb
import numpy as np

from . import __version__

logger = logging.getLogger(__name__)


# Valid inventory state transitions
VALID_TRANSITIONS = {
    'in_transit': {'received'},
    'received': {'saleable', 'damaged'},
    'saleable': {'committed', 'in_transit', 'damaged'},
    'committed': {'shipped'},
    'damaged': {'disposed', 'saleable'},
}
TERMINAL_STATES = {'shipped', 'disposed'}


class DrawdownEngine:
    """Time-stepped simulation engine for drawdown scenarios.

    Processes all events for time step T before advancing to T+1.
    No ordering or replenishment decisions — just consumption of
    pre-loaded inventory and arrival of pre-scheduled inbound shipments.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection, scenario_id: str):
        self.conn = conn
        self.scenario_id = scenario_id
        self._event_counter = 0

        # Load scenario config
        row = conn.execute(
            "SELECT * FROM scenario WHERE scenario_id = ?", [scenario_id]
        ).fetchone()
        if not row:
            raise ValueError(f"Scenario not found: {scenario_id}")

        cols = [desc[0] for desc in conn.description]
        self.scenario = dict(zip(cols, row))

        self.dataset_version_id = self.scenario['dataset_version_id']
        self.start_date = self.scenario['start_date']
        self.end_date = self.scenario['end_date']
        self.backorder_prob = float(self.scenario['backorder_probability'])
        self.write_event_log = self.scenario['write_event_log']
        self.write_snapshots = self.scenario['write_snapshots']
        self.snapshot_interval = self.scenario['snapshot_interval_days']

        # RNG for stochastic elements (backorder/lost-sale coin flip)
        self.rng = np.random.default_rng(42)

        # In-memory inventory state: {(dist_node_id, product_id, state): quantity}
        self._inventory: Dict[Tuple[str, str, str], float] = {}

        # Backorder queue: [(demand_id, demand_node_id, product_id, quantity, sim_date)]
        self._backorders: List[Tuple] = []

        # Buffered event log rows for batch insert
        self._event_buffer: List[Tuple] = []
        self._snapshot_buffer: List[Tuple] = []

        # Pre-load edge routing info for fulfillment
        self._fulfillment_routes: Dict[str, List[Tuple[str, str, float]]] = {}
        # Maps demand_node_id -> [(dist_node_id, edge_id, variable_cost)] sorted by cost

    def run(self):
        """Execute the full simulation."""
        start_time = time.time()
        logger.info(f"Starting simulation: {self.scenario_id}")
        logger.info(f"  Period: {self.start_date} to {self.end_date}")

        # Record run start
        self.conn.execute("""
            INSERT OR REPLACE INTO run_metadata
            (scenario_id, run_started_at, status, engine_version, config_snapshot)
            VALUES (?, ?, 'running', ?, ?)
        """, [
            self.scenario_id, datetime.now(), __version__,
            json.dumps({k: str(v) for k, v in self.scenario.items()}),
        ])

        try:
            self._initialize_inventory()
            self._build_fulfillment_routes()
            self._run_time_steps()

            elapsed = time.time() - start_time
            total_steps = self._compute_total_steps()

            self.conn.execute("""
                UPDATE run_metadata
                SET run_completed_at = ?, status = 'completed',
                    total_steps = ?, wall_clock_seconds = ?
                WHERE scenario_id = ?
            """, [datetime.now(), total_steps, round(elapsed, 2), self.scenario_id])

            logger.info(f"Simulation completed in {elapsed:.2f}s ({total_steps} steps)")

        except Exception as e:
            elapsed = time.time() - start_time
            self.conn.execute("""
                UPDATE run_metadata
                SET run_completed_at = ?, status = 'failed',
                    wall_clock_seconds = ?, error_message = ?
                WHERE scenario_id = ?
            """, [datetime.now(), round(elapsed, 2), str(e), self.scenario_id])
            logger.error(f"Simulation failed: {e}")
            raise

    def _compute_total_steps(self) -> int:
        d = self.start_date
        if isinstance(d, str):
            d = date.fromisoformat(d)
        e = self.end_date
        if isinstance(e, str):
            e = date.fromisoformat(e)
        return (e - d).days + 1

    def _initialize_inventory(self):
        """Load initial inventory from the database into memory."""
        rows = self.conn.execute("""
            SELECT dist_node_id, product_id, inventory_state, quantity
            FROM initial_inventory
            WHERE dataset_version_id = ?
        """, [self.dataset_version_id]).fetchall()

        for dist_node_id, product_id, state, qty in rows:
            key = (dist_node_id, product_id, state)
            self._inventory[key] = float(qty)

        total_units = sum(self._inventory.values())
        logger.info(f"Initialized inventory: {len(rows)} positions, {total_units:.0f} total units")

    def _build_fulfillment_routes(self):
        """Build routing tables: for each demand node, which distribution nodes can fulfill.

        Sorted by lowest variable cost first (simple drawdown routing).
        """
        rows = self.conn.execute("""
            SELECT e.edge_id, e.origin_node_id, e.dest_node_id, e.cost_variable
            FROM edge e
            WHERE e.dest_node_type = 'demand' AND e.origin_node_type = 'distribution'
            ORDER BY e.cost_variable ASC
        """).fetchall()

        for edge_id, origin_id, dest_id, cost_var in rows:
            if dest_id not in self._fulfillment_routes:
                self._fulfillment_routes[dest_id] = []
            self._fulfillment_routes[dest_id].append(
                (origin_id, edge_id, float(cost_var or 0))
            )

        logger.info(f"Built fulfillment routes for {len(self._fulfillment_routes)} demand nodes")

    def _run_time_steps(self):
        """Main simulation loop: iterate day by day."""
        current = self.start_date
        if isinstance(current, str):
            current = date.fromisoformat(current)
        end = self.end_date
        if isinstance(end, str):
            end = date.fromisoformat(end)

        step = 0
        while current <= end:
            self._process_day(current, step)
            step += 1
            current += timedelta(days=1)

        # Flush any remaining buffered events
        self._flush_events()
        self._flush_snapshots()

    def _process_day(self, sim_date: date, sim_step: int):
        """Process all events for a single day."""

        # 1. Process inbound arrivals
        self._process_inbound_arrivals(sim_date, sim_step)

        # 2. Process received -> saleable transitions
        self._process_receiving(sim_date, sim_step)

        # 3. Try to fulfill backorders first (FIFO)
        self._process_backorders(sim_date, sim_step)

        # 4. Process new demand for this day
        self._process_demand(sim_date, sim_step)

        # 5. Record daily fixed costs for distribution nodes
        self._record_fixed_costs(sim_date, sim_step)

        # 6. Write inventory snapshot (if enabled and on schedule)
        if self.write_snapshots and sim_step % self.snapshot_interval == 0:
            self._write_snapshot(sim_date)

        # Periodically flush event buffer
        if len(self._event_buffer) >= 5000:
            self._flush_events()

    def _process_inbound_arrivals(self, sim_date: date, sim_step: int):
        """Process scheduled inbound shipments arriving today."""
        rows = self.conn.execute("""
            SELECT inbound_id, supply_node_id, dest_node_id, product_id, quantity
            FROM inbound_schedule
            WHERE dataset_version_id = ? AND arrival_date = ?
        """, [self.dataset_version_id, sim_date]).fetchall()

        for inbound_id, supply_node_id, dest_node_id, product_id, qty in rows:
            qty = float(qty)

            # Add to received inventory
            key = (dest_node_id, product_id, 'received')
            self._inventory[key] = self._inventory.get(key, 0) + qty

            self._log_event(sim_date, sim_step, 'shipment_arrived',
                            node_id=dest_node_id, node_type='distribution',
                            product_id=product_id, quantity=qty,
                            from_state='in_transit', to_state='received',
                            detail=json.dumps({'inbound_id': inbound_id,
                                               'supply_node_id': supply_node_id}))

    def _process_receiving(self, sim_date: date, sim_step: int):
        """Transition received inventory to saleable.

        For simplicity in Phase 1, all received inventory becomes saleable
        on the same day (order_response_time applies to outbound, not receiving).
        """
        received_keys = [k for k in self._inventory
                         if k[2] == 'received' and self._inventory[k] > 0]

        for key in received_keys:
            dist_node_id, product_id, _ = key
            qty = self._inventory[key]

            # Move received -> saleable
            self._inventory[key] = 0
            saleable_key = (dist_node_id, product_id, 'saleable')
            self._inventory[saleable_key] = self._inventory.get(saleable_key, 0) + qty

            self._log_event(sim_date, sim_step, 'inventory_state_change',
                            node_id=dist_node_id, node_type='distribution',
                            product_id=product_id, quantity=qty,
                            from_state='received', to_state='saleable')

    def _process_backorders(self, sim_date: date, sim_step: int):
        """Attempt to fulfill backorders from available inventory (FIFO)."""
        remaining_backorders = []

        for demand_id, demand_node_id, product_id, qty, original_date in self._backorders:
            fulfilled_qty = self._try_fulfill(
                sim_date, sim_step, demand_id, demand_node_id,
                product_id, qty, is_backorder=True
            )
            unfulfilled = qty - fulfilled_qty
            if unfulfilled > 0:
                remaining_backorders.append(
                    (demand_id, demand_node_id, product_id, unfulfilled, original_date)
                )

        self._backorders = remaining_backorders

    def _process_demand(self, sim_date: date, sim_step: int):
        """Process all demand events for this day."""
        rows = self.conn.execute("""
            SELECT demand_id, demand_node_id, product_id, quantity
            FROM demand
            WHERE dataset_version_id = ? AND demand_date = ?
            ORDER BY demand_datetime ASC NULLS LAST
        """, [self.dataset_version_id, sim_date]).fetchall()

        for demand_id, demand_node_id, product_id, qty in rows:
            qty = float(qty)

            # Log demand received
            self._log_event(sim_date, sim_step, 'demand_received',
                            node_id=demand_node_id, node_type='demand',
                            product_id=product_id, quantity=qty,
                            demand_id=demand_id)

            # Attempt fulfillment
            fulfilled_qty = self._try_fulfill(
                sim_date, sim_step, demand_id, demand_node_id,
                product_id, qty
            )

            unfulfilled = qty - fulfilled_qty
            if unfulfilled > 0:
                self._handle_unfulfilled(
                    sim_date, sim_step, demand_id, demand_node_id,
                    product_id, unfulfilled
                )

    def _try_fulfill(self, sim_date: date, sim_step: int, demand_id: str,
                     demand_node_id: str, product_id: str, qty: float,
                     is_backorder: bool = False) -> float:
        """Try to fulfill demand from distribution nodes connected to this demand node.

        Returns the quantity successfully fulfilled.
        """
        routes = self._fulfillment_routes.get(demand_node_id, [])
        if not routes:
            return 0.0

        total_fulfilled = 0.0
        remaining = qty

        for dist_node_id, edge_id, variable_cost in routes:
            if remaining <= 0:
                break

            saleable_key = (dist_node_id, product_id, 'saleable')
            available = self._inventory.get(saleable_key, 0)
            if available <= 0:
                continue

            fulfill_qty = min(remaining, available)

            # Deduct from saleable
            self._inventory[saleable_key] -= fulfill_qty

            # Add to committed (then immediately shipped for drawdown simplicity)
            # In Phase 1, we collapse committed -> shipped into one step
            shipped_key = (dist_node_id, product_id, 'shipped')
            self._inventory[shipped_key] = self._inventory.get(shipped_key, 0) + fulfill_qty

            # Calculate cost
            cost = fulfill_qty * variable_cost

            event_type = 'backorder_fulfilled' if is_backorder else 'demand_fulfilled'
            self._log_event(sim_date, sim_step, event_type,
                            node_id=dist_node_id, node_type='distribution',
                            edge_id=edge_id, product_id=product_id,
                            quantity=fulfill_qty,
                            from_state='saleable', to_state='shipped',
                            demand_id=demand_id, cost=cost,
                            detail=json.dumps({
                                'demand_node_id': demand_node_id,
                                'variable_cost_per_unit': variable_cost,
                            }))

            total_fulfilled += fulfill_qty
            remaining -= fulfill_qty

        return total_fulfilled

    def _handle_unfulfilled(self, sim_date: date, sim_step: int,
                            demand_id: str, demand_node_id: str,
                            product_id: str, qty: float):
        """Handle demand that couldn't be fulfilled: backorder or lost sale."""
        if self.rng.random() < self.backorder_prob:
            # Backorder
            self._backorders.append(
                (demand_id, demand_node_id, product_id, qty, sim_date)
            )
            self._log_event(sim_date, sim_step, 'demand_backordered',
                            node_id=demand_node_id, node_type='demand',
                            product_id=product_id, quantity=qty,
                            demand_id=demand_id)
        else:
            # Lost sale
            self._log_event(sim_date, sim_step, 'demand_lost',
                            node_id=demand_node_id, node_type='demand',
                            product_id=product_id, quantity=qty,
                            demand_id=demand_id)

    def _record_fixed_costs(self, sim_date: date, sim_step: int):
        """Record daily fixed costs for active distribution nodes."""
        rows = self.conn.execute("""
            SELECT dist_node_id, fixed_cost, fixed_cost_basis
            FROM distribution_node
            WHERE fixed_cost > 0
        """).fetchall()

        for dist_node_id, fixed_cost, basis in rows:
            if basis == 'per_day':
                self._log_event(sim_date, sim_step, 'capacity_overage',
                                node_id=dist_node_id, node_type='distribution',
                                cost=float(fixed_cost),
                                detail=json.dumps({
                                    'cost_type': 'fixed_cost',
                                    'basis': basis,
                                }))

    def _log_event(self, sim_date: date, sim_step: int, event_type: str,
                   node_id: str = None, node_type: str = None,
                   edge_id: str = None, product_id: str = None,
                   quantity: float = None, from_state: str = None,
                   to_state: str = None, demand_id: str = None,
                   cost: float = None, detail: str = None):
        """Buffer an event for batch insertion."""
        if not self.write_event_log:
            return

        self._event_counter += 1
        self._event_buffer.append((
            self.scenario_id, self._event_counter, sim_date, sim_step,
            event_type, node_id, node_type, edge_id, product_id,
            quantity, from_state, to_state, demand_id, cost, detail,
        ))

    def _flush_events(self):
        """Batch insert buffered events into DuckDB."""
        if not self._event_buffer:
            return

        self.conn.executemany("""
            INSERT INTO event_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, self._event_buffer)
        self._event_buffer.clear()

    def _write_snapshot(self, sim_date: date):
        """Write current inventory state as a snapshot."""
        # Get product cube data for total_cube calculation
        product_cubes = {}
        rows = self.conn.execute("SELECT product_id, cube, cube_uom FROM product").fetchall()
        for pid, cube, cube_uom in rows:
            product_cubes[pid] = (float(cube), cube_uom)

        for (dist_node_id, product_id, state), qty in self._inventory.items():
            if qty <= 0:
                continue
            if state in TERMINAL_STATES:
                continue  # Don't snapshot shipped/disposed

            cube_per_unit, cube_uom = product_cubes.get(product_id, (0, 'L'))
            total_cube = qty * cube_per_unit

            self._snapshot_buffer.append((
                self.scenario_id, sim_date, dist_node_id, product_id,
                state, qty, total_cube, cube_uom,
            ))

        if len(self._snapshot_buffer) >= 5000:
            self._flush_snapshots()

    def _flush_snapshots(self):
        """Batch insert buffered snapshots into DuckDB."""
        if not self._snapshot_buffer:
            return

        self.conn.executemany("""
            INSERT INTO inventory_snapshot VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, self._snapshot_buffer)
        self._snapshot_buffer.clear()
