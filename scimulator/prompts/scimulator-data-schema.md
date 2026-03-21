# SCimulator Data Schema v0.2

This document defines the DuckDB relational schema for the Distribution SCimulator. All tables are stored in DuckDB. See `scimulator-v0.2.md` for the full design specification.

## Conventions

* All IDs are TEXT (human-readable, user-defined)
* All monetary values are DECIMAL(12,4) in the scenario's configured currency
* All weights are in kg, volumes in liters, distances in km
* Timestamps are TIMESTAMP; dates are DATE
* Sparse storage: missing inventory rows imply zero quantity
* PK = Primary Key, FK = Foreign Key, NN = Not Null

---

## Table Summary

### Configuration / Input Tables
| Table | Description |
|-------|-------------|
| scenario | Top-level scenario definition |
| scenario_param | Key-value parameter overrides per scenario |
| dataset_version | Immutable named input dataset registry |
| supplier | Supplier entities |
| supply_node | Individual supply locations |
| supply_node_tag | Tags for supply nodes |
| supply_node_product | Products available at each supply node |
| distribution_node | Distribution/fulfillment locations |
| distribution_node_tag | Tags for distribution nodes |
| demand_node | Customer or customer-aggregation locations |
| edge | Transportation links between nodes |
| product | Product master |
| product_attribute | Extensible key-value attributes per product |
| distance_matrix | Pre-computed or imported distances between locations |
| zone_table | Carrier zone/speed tables (ZIP3-based) |

### Simulation Input Tables (per dataset version)
| Table | Description |
|-------|-------------|
| demand | Customer demand events |
| inbound_schedule | Pre-determined inbound shipments (drawdown phase) |
| initial_inventory | Starting inventory positions |

### Simulation Output Tables (per scenario run)
| Table | Description |
|-------|-------------|
| event_log | Every discrete simulation event |
| inventory_snapshot | End-of-period inventory by node/product/state |
| run_metadata | Run timing, status, configuration snapshot |

---

## Configuration Tables

### scenario
Top-level definition of a simulation run.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| scenario_id | TEXT | PK | Unique scenario identifier |
| name | TEXT | NN | Human-readable name |
| description | TEXT | | Free-text description |
| dataset_version_id | TEXT | FK → dataset_version | Input data version |
| currency_code | TEXT | NN, default 'USD' | ISO 4217 currency code |
| time_resolution | TEXT | NN, default 'daily' | 'daily' or 'hourly' |
| start_date | DATE | NN | Simulation start date |
| end_date | DATE | NN | Simulation end date |
| warm_up_days | INTEGER | NN, default 0 | Days before measurement begins |
| backorder_probability | DECIMAL(5,4) | NN, default 1.0 | Probability unfulfilled demand is backordered (vs. lost sale) |
| write_event_log | BOOLEAN | NN, default TRUE | Whether to write event log |
| write_snapshots | BOOLEAN | NN, default TRUE | Whether to write inventory snapshots |
| created_at | TIMESTAMP | NN | Creation timestamp |
| notes | TEXT | | User notes |

### scenario_param
Key-value parameter overrides. Allows scenarios to override any default without schema changes.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| scenario_id | TEXT | PK, FK → scenario | |
| param_key | TEXT | PK | Parameter name (dotted path, e.g. "node.FC_EAST.max_outbound") |
| param_value | TEXT | NN | Value (parsed by type at runtime) |

### dataset_version
Registry of immutable input datasets.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| dataset_version_id | TEXT | PK | Unique version identifier |
| name | TEXT | NN | Human-readable name |
| description | TEXT | | How this version was created |
| parent_version_id | TEXT | FK → dataset_version, nullable | Version this was derived from |
| created_at | TIMESTAMP | NN | Creation timestamp |
| created_by | TEXT | | 'user', 'ai_agent', 'import', etc. |

---

## Network Topology Tables

### supplier
Supplier entities (may own multiple supply nodes).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| supplier_id | TEXT | PK | Unique supplier identifier |
| name | TEXT | NN | Supplier name |
| default_lead_time_days | DECIMAL(6,2) | | Default order lead time |
| default_qty_reliability | DECIMAL(5,4) | default 1.0 | Default probability of shipping full quantity |
| default_timing_variance_days | DECIMAL(6,2) | default 0.0 | Default mean timing variance (positive = late) |
| timing_variance_distribution | TEXT | default 'normal' | Distribution type for timing variance |
| timing_variance_std_days | DECIMAL(6,2) | | Std dev for timing distribution |

### supply_node
Individual supply locations.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| supply_node_id | TEXT | PK | Unique node identifier |
| supplier_id | TEXT | FK → supplier | Parent supplier |
| name | TEXT | NN | Node name |
| latitude | DECIMAL(9,6) | | Location latitude |
| longitude | DECIMAL(9,6) | | Location longitude |
| lead_time_days | DECIMAL(6,2) | nullable | Override of supplier default |
| qty_reliability | DECIMAL(5,4) | nullable | Override of supplier default |
| timing_variance_days | DECIMAL(6,2) | nullable | Override of supplier default |
| timing_variance_distribution | TEXT | nullable | Override of supplier default |
| timing_variance_std_days | DECIMAL(6,2) | nullable | Override of supplier default |
| max_capacity_units_per_day | DECIMAL(12,2) | nullable | NULL = infinite capacity |

### supply_node_tag
Multi-valued tags for supply nodes.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| supply_node_id | TEXT | PK, FK → supply_node | |
| tag | TEXT | PK | Tag value (e.g. 'Asia', 'domestic') |

### supply_node_product
Products available at each supply node.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| supply_node_id | TEXT | PK, FK → supply_node | |
| product_id | TEXT | PK, FK → product | |
| unit_cost | DECIMAL(12,4) | nullable | Override of product standard cost |
| max_units_per_day | DECIMAL(12,2) | nullable | Product-specific capacity limit |

### distribution_node
Distribution, fulfillment, and warehousing locations.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| dist_node_id | TEXT | PK | Unique node identifier |
| name | TEXT | NN | Node name |
| latitude | DECIMAL(9,6) | | Location latitude |
| longitude | DECIMAL(9,6) | | Location longitude |
| storage_capacity_l | DECIMAL(14,2) | nullable | Max storage volume (liters). NULL = unlimited |
| max_inbound_units_per_day | DECIMAL(12,2) | nullable | NULL = unlimited |
| max_outbound_units_per_day | DECIMAL(12,2) | nullable | NULL = unlimited |
| max_outbound_orders_per_day | DECIMAL(12,2) | nullable | NULL = unlimited |
| order_response_time_days | DECIMAL(6,2) | NN, default 1.0 | Time from order to shipment |
| fixed_cost_per_day | DECIMAL(12,4) | default 0 | Fixed operating cost per day |
| variable_cost_per_unit | DECIMAL(12,4) | default 0 | Variable cost per unit shipped |
| variable_cost_per_order | DECIMAL(12,4) | default 0 | Variable cost per order shipped |
| variable_cost_per_l | DECIMAL(12,4) | default 0 | Variable cost per liter shipped |
| overage_penalty_per_unit_day | DECIMAL(12,4) | nullable | Penalty for exceeding capacity. NULL = use 2x variable_cost_per_unit |

### distribution_node_tag
Multi-valued tags for distribution nodes.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| dist_node_id | TEXT | PK, FK → distribution_node | |
| tag | TEXT | PK | Tag value (e.g. 'regional_fc', 'national_dc') |

### demand_node
Customer or customer-aggregation locations.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| demand_node_id | TEXT | PK | Customer ID or ZIP3 code (e.g. "Z017") |
| name | TEXT | NN | Display name |
| latitude | DECIMAL(9,6) | | Location latitude (ZIP3 centroid if aggregated) |
| longitude | DECIMAL(9,6) | | Location longitude |
| zip3 | TEXT | nullable | ZIP3 code (for zone-table lookups) |

### edge
Transportation links between nodes. Supports supply→dist, dist→dist, and dist→demand.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| edge_id | TEXT | PK | Unique edge identifier |
| origin_node_id | TEXT | NN | Source node ID (supply or distribution) |
| origin_node_type | TEXT | NN | 'supply' or 'distribution' |
| dest_node_id | TEXT | NN | Destination node ID (distribution or demand) |
| dest_node_type | TEXT | NN | 'distribution' or 'demand' |
| transport_type | TEXT | NN | 'parcel', 'air', 'tl', 'ltl', 'flex', 'multi_modal', 'ocean_dray' |
| mean_transit_time_days | DECIMAL(8,4) | NN | Mean transit time |
| transit_time_distribution | TEXT | default 'lognormal' | Distribution type for transit variability |
| transit_time_std_days | DECIMAL(8,4) | nullable | Std dev for transit time distribution |
| transit_time_skew | DECIMAL(8,4) | nullable | Skew parameter (for asymmetric distributions) |
| cost_model | TEXT | NN, default 'per_unit_distance' | Cost calculation method |
| cost_fixed | DECIMAL(12,4) | default 0 | Fixed cost per shipment |
| cost_per_unit | DECIMAL(12,4) | default 0 | Cost per unit |
| cost_per_kg_km | DECIMAL(12,6) | default 0 | Cost per kg per km |
| cost_per_l_km | DECIMAL(12,6) | default 0 | Cost per liter per km |
| distance_km | DECIMAL(10,2) | nullable | Pre-computed distance (NULL = compute at runtime) |
| distance_method | TEXT | nullable | 'haversine', 'driving', 'zone_table' |

*Note: Edges can be auto-generated from tag-based rules. The `edge` table stores the projected node-to-node result.*

---

## Product Tables

### product
Product master.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| product_id | TEXT | PK | Unique SKU identifier |
| name | TEXT | NN | Product name |
| standard_cost | DECIMAL(12,4) | NN | Standard unit cost |
| base_price | DECIMAL(12,4) | NN | Base selling price |
| weight_kg | DECIMAL(10,4) | NN | Weight per unit |
| cube_l | DECIMAL(10,4) | NN | Volume per unit (liters) |
| orderable_qty | INTEGER | NN, default 1 | Minimum orderable quantity |

### product_attribute
Extensible key-value attributes.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| product_id | TEXT | PK, FK → product | |
| attribute_key | TEXT | PK | Attribute name (e.g. 'brand', 'major_line') |
| attribute_value | TEXT | NN | Attribute value |

---

## Distance / Rate Tables

### distance_matrix
Pre-computed or imported pairwise distances.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| origin_id | TEXT | PK | Node ID or ZIP3 |
| dest_id | TEXT | PK | Node ID or ZIP3 |
| method | TEXT | PK | 'haversine', 'driving' |
| distance_km | DECIMAL(10,2) | NN | Distance |

### zone_table
Carrier zone and speed tables (ZIP3-based).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| carrier | TEXT | PK | Carrier name (e.g. 'UPS', 'FedEx') |
| service_level | TEXT | PK | Service (e.g. 'ground', 'express') |
| origin_zip3 | TEXT | PK | Origin ZIP3 |
| dest_zip3 | TEXT | PK | Destination ZIP3 |
| zone | INTEGER | NN | Zone number |
| transit_days | DECIMAL(4,1) | NN | Expected transit days |
| cost_per_lb | DECIMAL(8,4) | nullable | Per-pound rate |
| cost_per_dimweight_lb | DECIMAL(8,4) | nullable | Per dim-weight-pound rate |
| cost_base | DECIMAL(8,4) | nullable | Base charge |

---

## Simulation Input Tables

These tables are scoped to a `dataset_version_id`. A given scenario references one dataset version.

### demand
Customer demand events. Each row is a single-line order (one SKU). Multi-SKU orders will be linked by `order_id` in a future phase.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| dataset_version_id | TEXT | PK, FK → dataset_version | |
| demand_id | TEXT | PK | Unique demand event ID |
| demand_date | DATE | NN | Date (or datetime if hourly) |
| demand_datetime | TIMESTAMP | nullable | Exact timestamp (hourly resolution) |
| demand_node_id | TEXT | NN, FK → demand_node | Customer / ZIP3 node |
| product_id | TEXT | NN, FK → product | |
| quantity | DECIMAL(12,2) | NN | Units demanded |
| order_id | TEXT | nullable | Future: links multi-SKU order lines |

### inbound_schedule
Pre-determined inbound shipments for drawdown scenarios.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| dataset_version_id | TEXT | PK, FK → dataset_version | |
| inbound_id | TEXT | PK | Unique shipment ID |
| supply_node_id | TEXT | NN, FK → supply_node | Origin |
| dest_node_id | TEXT | NN, FK → distribution_node | Destination |
| product_id | TEXT | NN, FK → product | |
| quantity | DECIMAL(12,2) | NN | Units |
| ship_date | DATE | NN | Expected ship date |
| arrival_date | DATE | NN | Expected arrival date |

### initial_inventory
Starting inventory positions for simulation.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| dataset_version_id | TEXT | PK, FK → dataset_version | |
| dist_node_id | TEXT | PK, FK → distribution_node | |
| product_id | TEXT | PK, FK → product | |
| inventory_state | TEXT | PK | 'in_transit', 'received', 'saleable', 'committed', 'damaged' |
| quantity | DECIMAL(12,2) | NN | Units (non-zero only) |

---

## Simulation Output Tables

These tables are scoped to a `scenario_id` (one run per scenario).

### event_log
Append-only log of every discrete simulation event.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| scenario_id | TEXT | FK → scenario | |
| event_id | BIGINT | | Auto-incrementing within run |
| sim_date | DATE | NN | Simulation date |
| sim_step | INTEGER | NN | Time step number |
| event_type | TEXT | NN | See event types below |
| node_id | TEXT | nullable | Node where event occurred |
| node_type | TEXT | nullable | 'supply', 'distribution', 'demand' |
| edge_id | TEXT | nullable | Edge involved (for transit events) |
| product_id | TEXT | nullable, FK → product | |
| quantity | DECIMAL(12,2) | nullable | Units affected |
| from_state | TEXT | nullable | Inventory state before |
| to_state | TEXT | nullable | Inventory state after |
| demand_id | TEXT | nullable | Reference to demand event |
| cost | DECIMAL(12,4) | nullable | Cost incurred by this event |
| detail | TEXT | nullable | JSON blob for event-specific data |

**Event types**:
* `demand_received` — demand arrives at the system
* `demand_fulfilled` — demand shipped from a node
* `demand_backordered` — demand queued awaiting stock
* `demand_lost` — demand lost (unfulfillable, not backordered)
* `shipment_dispatched` — shipment leaves origin node
* `shipment_arrived` — shipment arrives at destination node
* `inventory_state_change` — inventory transitions between states (received→saleable, etc.)
* `capacity_overage` — node exceeded a capacity limit
* `backorder_fulfilled` — previously backordered demand is fulfilled

### inventory_snapshot
End-of-period inventory positions. Sparse: only non-zero rows.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| scenario_id | TEXT | FK → scenario | |
| sim_date | DATE | | |
| dist_node_id | TEXT | FK → distribution_node | |
| product_id | TEXT | FK → product | |
| inventory_state | TEXT | | 'in_transit', 'received', 'saleable', 'committed', 'damaged' |
| quantity | DECIMAL(12,2) | NN | Units on hand |
| cube_l | DECIMAL(14,2) | nullable | Total volume (quantity × product cube) |

*Composite key: (scenario_id, sim_date, dist_node_id, product_id, inventory_state)*

### run_metadata
Records execution details for each simulation run.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| scenario_id | TEXT | PK, FK → scenario | |
| run_started_at | TIMESTAMP | NN | Execution start |
| run_completed_at | TIMESTAMP | nullable | Execution end (NULL if still running or failed) |
| status | TEXT | NN | 'running', 'completed', 'failed' |
| total_steps | INTEGER | nullable | Total time steps processed |
| wall_clock_seconds | DECIMAL(10,2) | nullable | Elapsed time |
| error_message | TEXT | nullable | Error details if failed |
| engine_version | TEXT | nullable | Simulator version string |
| config_snapshot | TEXT | nullable | JSON dump of full resolved scenario config |

---

## Inventory State Machine Reference

Valid states and transitions, enforced by the simulation engine:

```
State             Allowed Transitions To
─────────────     ──────────────────────
in_transit        received
received          saleable, damaged
saleable          committed, in_transit, damaged
committed         shipped
shipped           (terminal)
damaged           disposed, saleable (via repair)
disposed          (terminal)
```

---

## Indexing Notes

Recommended indexes for query performance (DuckDB will auto-optimize, but these are logical access patterns):

* `event_log`: (scenario_id, sim_date), (scenario_id, event_type), (scenario_id, product_id)
* `inventory_snapshot`: (scenario_id, sim_date), (scenario_id, dist_node_id)
* `demand`: (dataset_version_id, demand_date), (dataset_version_id, demand_node_id)
* `edge`: (origin_node_id), (dest_node_id), (transport_type)
