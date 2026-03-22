# Distribution SCimulator — Design Specification v0.2

## 1. Overview

The Distribution SCimulator is a distribution network simulator that models product flows from suppliers to customers through N distribution layers (where N ranges from 0 to 4). It is designed for scenario-based exploration of network configurations, inventory policies, and fulfillment strategies.

### 1.1 Scale Targets

* Thousands of customers (optionally named, or aggregated to ZIP3 regions)
* Tens of thousands of products (SKUs)
* Hundreds to thousands of suppliers
* Up to 4 distribution layers
* Daily or hourly time resolution, for simulation periods up to 2–3 years

### 1.2 Performance Targets

For a mid-size scenario (5,000 customers, 10,000 SKUs, 500 suppliers, 3 distribution layers, 1 year daily):
* Drawdown (no decision logic): a few seconds
* With ordering and fulfillment logic: 1–2 minutes

These targets make Python with vectorization (NumPy/Numba) the primary implementation language. If performance proves insufficient, Julia or Rust may be considered for the simulation core.

### 1.3 Simulation Approach

The simulator uses a **time-stepped** architecture: all events for time step T are processed before advancing to T+1. This enables vectorized operations across nodes, products, and orders within each step. When decision logic triggers future events (e.g., a purchase order creates a future arrival), those events are pre-loaded into the appropriate future time step.

Time resolution is a **per-scenario setting** (daily or hourly). In the initial phase, input data granularity must match the scenario time step. Future phases will support aggregation of finer-grained data into coarser time steps and sub-daily ordering (e.g., carrier cut-off times).

### 1.4 Design Principles

* **Modularity**: Every component (fulfillment logic, ordering policy, cost models, distance calculations) should be a swappable module. The system will grow in complexity over time; modular design prevents rewrites.
* **Scenario-centric**: A scenario fully defines a simulation run. Even in early phases, the scenario model should be comprehensive enough to avoid refactoring later.
* **Sparse by default**: Only non-zero values are stored. Missing inventory records imply zero.

## 2. Application Infrastructure

### 2.1 Architecture

Client-server application:
* **Server**: Python (FastAPI), running locally on Mac/Linux initially, later in a Linux Docker container for flexible deployment.
* **Client**: Modern web browser (Chrome). Mobile-responsive UI for phone and tablet.
* **Database**: DuckDB from day one. File-based, zero-server-overhead, excellent Python integration, SQL query capability over all simulation data. Replaces the need for separate Parquet files.

### 2.2 Simulation Engine Isolation

The simulation engine is **stateless**. All state lives in DuckDB. The engine accepts a `(scenario_config, db_path)` tuple and writes results to the database. This makes parallelism trivial: each run gets its own DuckDB file or schema.

### 2.3 Units of Measure

Every measured value in the schema has an explicit UoM column (e.g. `storage_capacity` + `storage_capacity_uom`). This prevents ambiguity and allows mixed units where natural (e.g. liters for product cube, cubic meters for warehouse capacity). A `uom` reference table defines valid units with conversion factors, enabling the engine to validate and auto-convert.

Default metric units by dimension:

| Dimension | Default Unit | Common Alternatives |
|-----------|-------------|---------------------|
| Mass      | kg          | lb, g               |
| Volume    | L (product), m3 (facility) | cuft, gal |
| Length    | cm          | in                  |
| Distance  | km          | mi                  |
| Time      | days        | hours               |

Cost fields use a `_basis` column instead of a UoM, describing what the cost is per (e.g. `per_unit`, `per_m3`, `per_order`, `pct_value`). This supports flexible cost structures — the same `variable_cost` field can represent per-unit, per-volume, or percentage-of-value costs depending on the basis.

The UI supports display in both metric and imperial units.

### 2.4 Currency

One currency per scenario (configurable — USD, EUR, etc.). Multi-currency with exchange rates is a future enhancement.

## 3. Network Entities

The distribution network is defined by four entity types plus the logic that governs their interactions.

### 3.1 Demand Nodes

Demand nodes represent customers or customer aggregations. They are the terminal sinks of product flow.

* **Granularity**: Arbitrary — from individual named customers (e.g., "Customer #1234") to geographic aggregations (e.g., ZIP3 "Z017"). When aggregating, the ZIP3 code serves as the customer identifier.
* **Dual role**: A retail storefront can be modeled as either a demand node (terminal sink) or a distribution node (intermediate), depending on the question being addressed.
* **Demand data**: The primary input. Synthetic demand can be generated by the Synthetic Demand Engine (located at `/scimulator/synthetic_demand_engine`), which produces discrete order-level output with geographic weights by ZIP3.

### 3.2 Supply Nodes

Supply nodes are product sources entering the network. They may represent factories, overseas warehouses, or any entity outside the simulated network boundary.

Key attributes:
* **Tags**: Flexible, multi-valued type tags used by policy rules (e.g., "Asia", "domestic", "preferred")
* **Product catalog**: Which products the node can supply
* **Capacity**: Units of product per time period (initially assumed infinite)
* **Lead time**: Order-to-ship time
* **Location**: Lat/lon coordinates
* **Reliability**:
    * *Quantity reliability*: Probability of shipping in full; possibility of over-shipment
    * *Timing reliability*: Distribution of ship-date variance (late and early). This is a key characteristic to model.
* **Supplier grouping**: A supplier entity may have multiple supply nodes. Attributes can be defined at the supplier level and overridden per supply node.

### 3.3 Distribution Nodes

Intermediate nodes that receive, store, and ship products. May be arranged in multiple tiers with replenishment and rebalancing between them.

Key attributes:
* **Tags**: Arbitrary, multi-valued type tags (e.g., "retail_store", "regional_fc", "national_dc"). Tags drive policy application. A single node can have multiple tags to represent co-located functions sharing capacity.
* **Storage capacity**: Volume with explicit UoM (default m3 for facilities)
* **Inbound capacity**: Per day, with explicit UoM (units, m3, etc.)
* **Outbound capacity**: Per day, with explicit UoM (units, orders, m3, etc.)
* **Order response time**: Time from order placement to shipment, with UoM (node-level initially; future: per node-product)
* **Fixed cost**: With explicit basis (default: per day)
* **Variable cost**: With explicit basis (e.g., per_unit, per_order, per_m3, pct_value)
* **Outbound modes**: Supported transport types (TL, LTL, parcel, flex)
* **Overage penalty cost**: With explicit basis; defaults to 2x variable cost if not specified

#### Capacity Enforcement

Capacity uses **soft constraints**: overages are allowed but incur penalty costs. This ensures the simulation never breaks due to overflow and capacity problems surface naturally in cost KPIs. Future enhancement: configurable hard-cap multiplier (e.g., reject at 1.5x capacity) per constraint type.

### 3.4 Transportation Edges

Edges define which nodes can ship to which, along with cost and transit-time characteristics.

Key attributes:
* **Transport type**: Parcel, Air, TL, LTL, flex, multi-modal, ocean + dray
* **Origin(s) and destination(s)**
* **Transit time**: Mean time (minutes, hours, or days), with asymmetric variability distribution and configurable parameters
* **Cost structure** (examples):
    * Zone and dim-weight based (parcel)
    * Fixed cost + per-cube-distance or per-weight-distance
    * LTL cost structures (future)

#### Edge Definition Granularity

Edges can be defined at multiple levels:
* **Specific**: "Supplier A ships to nodes B, C, D"
* **Tag-based**: "All supply nodes tagged 'Asia' can ship to all regional FCs tagged 'US' or 'Canada' via ocean"
* **Mixed**: "Regional FC A ships to demand nodes B–M via TL, and nodes N–Z via air or ground parcel"

Higher-level rules are projected into node-to-node edges before simulation.

#### Distance Calculation

Three modes supported:
* **Haversine**: Great-circle distance with configurable circuity factor
* **Driving distance**: Point-to-point road distance (e.g., for TL shipments)
* **Zone tables**: ZIP3-based zone and speed tables (e.g., UPS/FedEx rate structures)

## 4. Products

### 4.1 Product Master

Each product (SKU) has core attributes:

| Attribute     | Description                    |
|---------------|--------------------------------|
| product_id    | Unique SKU identifier          |
| cost          | Standard unit cost             |
| base_price    | Base selling price             |
| weight_kg     | Weight per unit (kg)           |
| cube_l        | Volume per unit (liters)       |
| orderable_qty | Minimum orderable quantity     |

### 4.2 Product Attributes

An extensible key-value attribute table supports arbitrary additional attributes per SKU:
* Brand
* Major product line
* Minor product line
* Custom attributes as needed

### 4.3 Future Product Features

* **Supersessions**: Product replacement chains
* **Kits**: Composite products assembled from component SKUs

## 5. Orders

An **order** is defined as: one event, one customer, one or more SKUs, each with a quantity (which may vary by SKU).

* Orders can be **split by SKU** across fulfillment nodes (common).
* Orders can be **split by quantity** for a single SKU (edge case, but supported).
* In the initial phase, each demand engine row is treated as a single-line order. Multi-SKU orders (bundled by customer + day) are a future enhancement, using an explicit order ID.

### 5.1 Unfulfillable Demand

When demand cannot be fulfilled from stock, the outcome is governed by a **backorder probability** parameter (per scenario):
* With probability P: demand is backordered (deferred until inventory arrives)
* With probability 1-P: demand is a lost sale (evaporates permanently)

Future enhancement: probability that varies with expected delivery promise time.

## 6. Inventory

### 6.1 Inventory Tracking

Inventory is tracked at the granularity of: **product × location × state**, where location is a node or an in-transit edge.

### 6.2 Inventory States

Inventory follows a defined state machine:

```
in_transit → received → saleable → committed → shipped
                     → damaged  → disposed
                                → repaired → saleable
saleable   → in_transit  (inter-node transfers)
```

States:
* **in_transit**: Moving on a transportation edge
* **received**: Arrived at node, not yet available (processing time)
* **saleable**: On shelf, available for fulfillment or transfer
* **committed**: Reserved for a customer order, in the shipping process
* **shipped**: Left the node, en route to customer (terminal state at the node)
* **damaged**: In stock but not saleable
* **disposed**: Removed from inventory (terminal state)
* **repaired**: Returned to saleable condition

Future extensions: quarantine, returned.

### 6.3 Inventory Storage

Sparse storage: only non-zero inventory is recorded. A missing row for a (day, node, SKU, state) combination implies zero inventory. DuckDB columnar compression handles this efficiently. Snapshots may be partitioned by simulation month for query performance.

## 7. Network Decision-Making

Decision logic is deferred to Phase 3 (Intelligence). The initial implementation supports drawdown only: pre-loaded inventory and pre-determined demand/inbound events, with no active decisions.

Key decisions to support in Phase 3+:
* **Supplier ordering**: How much to order, when, where to ship
* **Cross-docking / re-routing**: Diversion decisions at time of receipt
* **Replenishment / rebalancing**: Inventory transfers between distribution nodes
* **Transport mode selection**: Choosing among available modes per edge
* **Demand forecasting**: Input to ordering and allocation decisions

All decision logic will be implemented as **swappable policy modules** (plugin pattern), so different strategies can be compared across scenarios.

## 8. Scenarios

A scenario fully defines a simulation run. This model is established from day one, even though many fields are unused in early phases.

### 8.1 Scenario Definition

* **Scenario ID and name**
* **Input dataset version**: Reference to an immutable, named dataset (demand, inbound schedules, initial inventory)
* **Entity sets**: Named subsets of network entities that participate in the scenario. One optional set per entity type:
    * `product_set_id` — which products to include
    * `supply_node_set_id` — which supply nodes to include
    * `distribution_node_set_id` — which distribution nodes to include
    * `demand_node_set_id` — which demand nodes (customers) to include
    * `edge_set_id` — which edges to include (further restricted by active node sets)
    * NULL for any set means "use all entities of that type"
* **Parameter overrides**: Backorder probability, node capacities, penalty costs, etc.
* **Policy rules**: Fulfillment logic, routing rules, ordering policies (Phase 3+)
* **Time settings**:
    * `time_resolution`: "daily" or "hourly"
    * `start_date`, `end_date`
    * `warm_up_days`: Integer; simulation days before measurement begins (ignored in drawdown phase, used with ordering logic)
* **Currency**: ISO currency code (e.g., "USD", "EUR")
* **Output settings**:
    * `write_event_log`: Boolean
    * `write_snapshots`: Boolean (can be disabled if storage is a concern)
    * `snapshot_interval_days`: Integer, default 1. Controls how often inventory snapshots are written (1 = daily, 7 = weekly, etc.). Any point-in-time state can be reconstructed from the nearest prior snapshot plus event log replay.

### 8.2 Entity Sets

Network entities (suppliers, supply nodes, distribution nodes, demand nodes, edges, products) are stored globally in the database — not scoped to a dataset version. **Entity sets** are named, static subsets that control which entities participate in a given scenario. Sets are defined via explicit membership (a header table + member table per entity type). Dynamic set definitions (e.g., "all supply nodes tagged 'Asia'") are a future feature.

When the engine resolves a scenario's active network:
* Edges referencing nodes outside the active node sets are pruned (with a warning logged).
* Nodes with no remaining edges are flagged as stranded (with a warning logged).
* Simulation input rows (demand, initial_inventory, inbound_schedule) are filtered against the active sets — rows referencing entities outside the sets are silently dropped.

This design means different scenarios can select different slices of the same entity data without duplication. For example, two variants of a distribution node (e.g., an FC at different capacity levels) are modeled as two separate entities, and sets choose which one to include.

### 8.3 Data Versioning

Simulation input data (demand, inbound schedules, initial inventory) is stored as **immutable, named versions** in DuckDB. Each version gets a unique ID, and scenarios reference dataset versions. New versions are created via copy-and-modify (e.g., AI agent: "increase demand by 10% for New England, March–May" creates a new dataset version). Original data is always preserved.

| Mechanism | Applies to | Purpose |
|-----------|-----------|---------|
| **Entity sets** | suppliers, supply_nodes, distribution_nodes, demand_nodes, edges, products | Which entities participate in the scenario |
| **Dataset versions** | demand, initial_inventory, inbound_schedule | Which variant of the simulation input data to use |

## 9. Simulation Outputs

### 9.1 Event Log

Every discrete event is recorded as a row: shipments, receipts, fulfillments, backorders, lost sales, inventory state transitions, capacity overages. Append-only in DuckDB. This is the granular record for drill-down analysis and debugging.

### 9.2 State Snapshots

Periodic inventory positions by node/product/state. Sparse (non-zero only). Snapshot frequency is configurable via `snapshot_interval_days` (default: daily). For less frequent snapshots, any specific day's state can be reconstructed from the nearest prior snapshot plus event log replay. Writing snapshots can be disabled entirely if storage is a concern.

### 9.3 Key Performance Indicators

All KPIs are queryable at any level of aggregation: overall, by node, edge, supplier, product, month, etc., thanks to the granular event log and snapshot data.

**Supplier Performance**
* On-time % (shipped, received)
* Median variance (days) in arrival date
* Overall and by supplier

**Fulfillment Performance**
* % of demand fulfilled from in-stock inventory
* % of demand with 1, 2, 3, 4+ day order-to-delivery (O2D)
* Median and average delivery days
* Variance from expected delivery (future: requires delivery speed variability modeling)
* % fulfilled from lowest-cost node
* % fulfilled from fastest node

**Inventory Performance**
* Average inventory levels
* Months of supply (inventory / sales)
* Inventory turns

**Costs** (reported as $, $/unit, and % of sales)
* Transport: inbound, replenishment, rebalancing, outbound
* Warehouse: fixed, variable
* Inventory carrying costs
* Capacity overage penalties

## 10. User Interface

### 10.1 Web GUI

Client-server web application (FastAPI backend, modern frontend framework). The client handles:
* Data upload (files or URLs)
* Configuration editing
* Simulation execution with real-time progress
* Visualization (charts, maps, network flow diagrams)
* KPI dashboards
* Data and image export

A flow visualizer proof of concept exists in `../flow_viz` (prompt: `./prompt_flow_viz.md`).

### 10.2 Mobile Interface

Simplified responsive UI for phone and tablet, supporting the same core workflows on smaller screens.

### 10.3 AI Agent Interface

An optional LLM-powered interface allows users to:
* Modify simulation configuration via natural language (e.g., "increase demand by 10% for all customers in New England, March–May")
* Query results conversationally (e.g., "how much did turns change between scenarios B and C?")
* Generate new dataset versions from natural-language instructions

### 10.4 Data Import/Export

Easy import and export of input and output data. Planned for a later phase.

## 11. Data Storage

All data is stored in **DuckDB**. The detailed data schema is specified in `./scimulator-data-schema.md`.

Key storage decisions:
* Sparse inventory storage (non-zero only; missing = zero)
* Columnar compression for efficient queries
* Snapshot partitioning by simulation month
* Immutable dataset versioning for inputs
* Append-only event logs for outputs

## 12. Development Phases

Each phase produces a working, testable system. Each phase avoids rework of prior phases.

### Phase 1: Foundation
1. DuckDB schema and data model (nodes, edges, products, inventory states)
2. Scenario configuration model
3. Demand engine integration (load demand into DuckDB)
4. Basic simulation loop (time-stepped, drawdown only)
5. Event log and snapshot output to DuckDB
6. CLI to run a scenario and dump results

### Phase 2: Visibility
7. Web UI scaffold (FastAPI + frontend)
8. Results dashboard (KPI tables, inventory charts)
9. Network visualization (adapt flow_viz)
10. Import/export (upload inputs, download results)
11. Scenario save/reload

### Phase 3: Intelligence
12. Order fulfillment logic (routing, node selection)
13. Supplier ordering logic (reorder points, forecasting)
14. Transportation mode selection
15. Backorder/lost-sale handling

### Phase 4: Scale
16. Batch runs (parallel scenario execution)
17. Data versioning (immutable datasets)
18. Scenario comparison views (KPI diff tables, map overlays)

### Phase 5: Polish
19. AI agent interface
20. Mobile UI
21. Docker deployment
22. Advanced features (cross-docking, rebalancing, supersessions, kits)

### Architectural Decisions Made Early (regardless of phase)
* Scenario/config schema — defined in Phase 1, used by all phases
* Event log format — consistent from Phase 1 onward
* Plugin/policy pattern — fulfillment and ordering logic are swappable modules

## 13. Open Design Issues

* **Implementation language**: Python+Numba is the starting point. Performance benchmarking in Phase 1 will determine if a faster language is needed for the simulation core.
* **Order and fulfillment logic**: Detailed design deferred to Phase 3. The plugin pattern ensures the drawdown engine doesn't need to change.
* **Sub-daily time resolution**: Daily/hourly mismatch handling (e.g., carrier cut-off times) is identified but deferred past Phase 1.
* **Steady-state initialization**: Simple warm-up period (configurable `warm_up_days`) for Phase 3. Heuristic pre-computation of starting inventory (average demand × target days-of-supply) is a tractable near-term enhancement.

## 14. Future Features Backlog

Consolidated list of all features mentioned as future enhancements throughout this spec. Grouped by domain, roughly ordered by expected priority within each group. Phase references indicate the earliest likely phase.

### Network & Topology
| Feature | Description | Earliest Phase |
|---------|-------------|----------------|
| LTL cost structures | LTL-specific rating models for transportation edges | Phase 3 |
| Configurable hard capacity caps | Hard-cap multiplier per constraint type (e.g., reject at 1.5x) alongside soft constraints | Phase 3 |
| Order response time per node-product | Granular processing time that varies by product at a given node, not just node-level | Phase 3 |
| Cross-docking / re-routing | Diversion decisions at time of receipt | Phase 5 |
| Inventory rebalancing | Transfers between distribution nodes to optimize stock positioning | Phase 5 |

### Products & Demand
| Feature | Description | Earliest Phase |
|---------|-------------|----------------|
| Multi-SKU orders | Bundle single-line demand into multi-SKU orders via explicit order ID | Phase 2 |
| Supersessions | Product replacement chains (old SKU → new SKU) | Phase 5 |
| Kits | Composite products assembled from component SKUs | Phase 5 |

### Simulation Logic
| Feature | Description | Earliest Phase |
|---------|-------------|----------------|
| Order fulfillment logic | Routing rules, node selection for fulfilling customer orders | Phase 3 |
| Supplier ordering logic | Reorder points, safety stock, demand forecasting | Phase 3 |
| Transportation mode selection | Choosing among available transport modes per edge | Phase 3 |
| Backorder probability by delivery promise | Lost-sale probability that increases with expected delivery time | Phase 3 |
| Delivery speed variability | Model variance in last-mile delivery for O2D variance KPI | Phase 3 |
| Demand forecasting | Forecast-driven ordering and allocation decisions | Phase 3+ |
| Steady-state initialization heuristic | Pre-compute starting inventory as avg demand × target days-of-supply | Phase 3 |

### Time Resolution
| Feature | Description | Earliest Phase |
|---------|-------------|----------------|
| Sub-daily ordering (carrier cut-offs) | Differentiate early vs. late orders within a day for outbound cut-off logic | Phase 3 |
| Data resolution mismatch handling | Aggregate hourly input data into daily steps (or vice versa) | Phase 2 |

### Data & Storage
| Feature | Description | Earliest Phase |
|---------|-------------|----------------|
| Multi-currency with exchange rates | Multiple currencies within a single scenario with conversion tables | Phase 4+ |
| Snapshot retention policies | Keep only last N days of snapshots instead of all-or-nothing | Phase 4 |
| Data import/export | Easy upload/download of input and output datasets | Phase 2 |
| Data versioning | Immutable named datasets with copy-and-modify via AI agent | Phase 4 |

### Scenarios & Scale
| Feature | Description | Earliest Phase |
|---------|-------------|----------------|
| Scenario save/reload | Persist and reload scenario configurations | Phase 2 |
| Batch scenario execution | Define and run many scenarios in parallel | Phase 4 |
| Scenario comparison views | KPI diff tables, map overlays comparing scenarios | Phase 4 |

### UI & Infrastructure
| Feature | Description | Earliest Phase |
|---------|-------------|----------------|
| AI agent interface | Natural language config changes and report queries | Phase 5 |
| Mobile UI | Simplified responsive interface for phone and tablet | Phase 5 |
| Docker deployment | Containerized deployment for flexible scaling | Phase 5 |

### Inventory States
| Feature | Description | Earliest Phase |
|---------|-------------|----------------|
| Quarantine state | Inventory held pending inspection | Phase 3+ |
| Returned state | Customer returns flowing back into inventory | Phase 3+ |
| Additional damage pathways | More transitions into the damaged state | Phase 3+ |
