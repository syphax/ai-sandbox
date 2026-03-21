# Preamble

To any agent reading this dcocument: If you have any questions, please add to the end of this document. OR, if you have been directed to fold any Q&A sections into the spec, please do so, so that the questions are sufficiently answered within the flow of the document.

# Distribution SCimulator

The scope of this effort is to build a distribution simulator that simulates flows from suppliers to customers through N distribution layers, where N can range from 0 to perhaps 4 layers.

Key considerations:

* We want to be able to simulate demand from 1000's of customers for 10's of thousands of products from hundreds-to-thousands of suppliers
* We want to simulate at a daily- and optionally hourly- time resolution, for periods up to 2-3 years
* We want flexibility, in terms of:
    * Defining network nodes and edges
    * Defining demand, products, supplier attributes
    * Defining routing logic
    * Defining inventory policies - customer order routing, re-order logic, etc.
    * Defining transportation costs and constraints
* THe model must be performant: We do not want to wait hours for results for a scenario. As we will want to explore many scenarios (across different supplier profiles, network configurations, routing logic, etc. etc.), performance is really important. Ideally, it would be fast enough so that building an optimization layer on top of this model would be feasible. We realize that this is a reach.
* That said, we prefer to build the model in Python, due to our familiarity with the language and the ecosystem of libraries available for data analysis, visualization, etc. etc. Other languages such as Julia or Rust could be considered if we can't make Python fast enough via vectorization and performance libraries like numba.

# Key UI considerations

## GUI

We will flesh this section more out later. This will be a web application. The client will be responsible for input and output. Most of the calculations will occur on the server side.

The client side will allow the user to:
* Upload data (or submit URLs for data)
* Edit configuration settings
* Run models
* View visualizations
* View output metrics
* Export images and data

We will need to provide much more detail here in an update to this document.

The client side will use modern web UI frameworks. One component of the UI will be the flow visualizer; we have already built a proof of concept in the `../flow_viz` directory (prompt is in `./prompt_flow_viz.md`)

We will want a mobile (phone and tablet) interface that is similar to the base UI, but simplified for smaller screens.

## AI

Also: we want to offer LLMs as an optional interface to the simulation configuration. In addition to editing tables and config files directly, we want to offer the capability for 

# Application Infrastructure

As stated, this will be a client-server application. The client will run in modern web browsers. The server will run, eventually, in a Linux docker container, so that we have flexibility to deploy and scale.

The model calculations can be in any language that is fast, readable, and can support the client-side UI. Python is a contender, especially with speed-ups like Numba, but we can also consider Julia, Rust, or other inherently faster languages. 

This is an open design issue.

Initially, we want to run directly on a local machine (assume a Mac or Linux machine with a Chrome browser), storing data in files. Once we have something mostly working, we want to port to a Docker container, for flexible deployment. 

# Network Entities

The type of distribution network we want to be able to simulate can be defined in terms of:
* Demand Nodes: Customers
* Supply Nodes: Suppliers
* Distribution Nodes: Distribution Centers, Warehouses, etc.
* Transportation Edges: Links between nodes, with associated costs and constraints
* Logic: The decisions that determine how demand should be fulfilled, distribution nodes get replenished, etc. 

## Demad Nodes

Demand nodes are customers, typically aggregated by geography. Most of the time, we will express demand in terms of dates and ZIP3 geographic areas. However, we don't want the model to rely on that level of resolution; we want the option of arbitrary granularity, down to the individual customer and order event.

A customer could be a retail customer, or it could be a demand "sink" like a store, where we don't need to simulate the performance of that entity- it just slurps up demand, case closed. So, in this simulator, something like a retail storefront could either be a demand node, or a distribution node, depending on what questions we are addressing.

One key input to the model is demand data. One source of synthetic demand data is Synthetic Demand Engine which can create a variety of demand sets.

## Supply Nodes

Supply nodes are treated as sources of products into the supply chain network. They may be factories, or warehouses that are considered outside the scope of the simulated supply chain network, apart from their role source nodes.

Fundamentally, products flow into the network from one or more supply nodes.

Supply nodes have some key attributes:
* Type: Flexible tag used by policy rules. Supply nodes can have multiple tags.
* What products they can supply
* Capacity limitations (e.g. can only supply X units of product Y in any given time period)- note, initially we will assume infinite capacity
* Order lead times
* Location.
* Order reliability. This has 2 components:
    * Quantity reliability- how often are orders shipped in full? Does the supplier ever over-ship?
    * Timing reliability- how often is the supply shipped late (or even early)? This is a key characteristic that we want to model!
* Supplier. A supplier may have more than one supply node. Supply node attributes can be defined at the suppler level, and overriden at the supply node level.

## Distribution Nodes

There are many possible types of distribution nodes- retail stores, local warehouses, regional fulfillment centers, national distribution centers. They can be arranged in many tiers, and may replenish or rebalance inventory between them, based on inventory management rules.

Key distribution node attributes:
* Type: we can support arbitrary types. Possible options could be: retail store, local fulfillment center, regional fulfillment center, national distribution center. These types are not just descriptive, as we use them to apply various policies. Important: a node can have more than one type! This is to support co-located functions that share space. So modelers could either define a located facility with two nodes of single types and fixed capacities, or one node with overall capacities.
* Size (storage cube, in either m3 or cuft)
* Max inbound capacity (cube or units per day)
* Max outbound capacity (cube or units or orders per day)
* Order response time (time from order placement to shipment)- this can be much more granular than node level (e.g. a function of node and product), but initially, this will be a node-level parameter
* Fixed cost
* Variable cost (per unit, order, or cube shipped)
* Outbound modes supported- e.g. TL, LTL, parcel, flex (flex = distributed shipping like Amazon Flex)

## Transportation Edges

"Edges" determine which nodes in the network can ship to which. At the extreme, any node could ship to any distribution or demand node. Some common sequences are:
* Supplier - national distribution center(s) - regional distribution center(s) - demand nodes
* Supplier - regional FC - regional FC - demand node

Key attributes of edges:
* Transport type (Parcel, Air, TL, LTL, flex, multi-model, ocean + dray)
* Origin(s) and destination(s)
* Mean transit time (minutes, hours or days)
    * Note: When we define general edge rules (e.g. all overseas suppliers can ship to all regional FCs), we will want this to be a function of distance
* Shape of variability (usually not symmetrical)
* Parameters for the selected distribution
* Cost structure. Examples:
    * Zone and dimweight based (parcel)
    * Fixed cost + per-cube-distance or per-weight-distance 
    * LTL cost structures (we won't tackle these yet)

We will want to be able to define edges at various levels of granularity, from very specific (supplier A can ship to distribution nodes B, C, and D), to general ("all supply nodes that are tagged "Asia" can ship to all regional FCs tagged "US" or "Canada" via ocean"). Or, "Regional FC A can ship to demand nodes B-M via TL, and nodes N-Z via air or ground parcel"

Ultimately, higher-level rules will be projected into node-node level rules that the simulation will use.

# Inventory

We need to track inventory at a detailed level- we need to know how much of each product is where; "where" can be a node, or traveling on an edge. And, inventory can have different "states"- received (but not available yet), saleable (received, on shelf, ready to fulfill orders or moves), committed (in-stock but committed to a customer, typically during the shipping process), damaged (in-stock but not saleable), etc.

# Network Decision-Making

This is a really important section, but we're actually going to skip over it for now, as our near-term objective is to do enable "bleed down" functionality from pre-determined demand and inbound moves. 

Key decisions we need to support:
* Orders from suppliers into the network- how much, when, where to ship?
* Re-routing (cross-docking) between nodes- this occurs at the time of receipt
* Replenishment or inventory balancing between nodes
* What transport mode to use for various moves

Some of these may involve more involved logic, like demand forecasting, etc. 

This is auper important part of the whole thing; we'll revisit later!

# Measures and Outputs

We want to retain a lot of flexibility on what report, which requires holding onto a lot of data. Fortunately, in the modern age, this is not a big deal.

We will need to flesh this section out more. There are many potential KPIs to track.

These include:

* Supplier performance
    * On-time % (shipped, received)
    * Med variance (days) in arrival date
    * Overall and by supplier

* Fulfillment performance
    * % of demand fulfilled from in-stock inventory
    * % of demand with 1, 2, 3, 4, etc. day O2D (order to delivery) days
    * Median and average delivery days
    * Future: Variance from expected delivery (this will require modeling variance in delivery speeds)
    * Pct. fulfilled from lowest cost node (outbound cost)
    * Pct. fulfilled from fastest node

* Inventory performance
    * Avg. inventory levels
    * Months of supply (inventory / sales from inventory)
    * Turns (inverse of MOS)

* Costs
    * $, $ / unit, % of sales
    * Transport
        * Inbound
        * Replanishment 
        * Rebalancing
        * Outbound
    * Warehouse (Fulfillment Nodes)
        * Fixed
        * Variable
    * Cost of inventory (carrying costs)

We will want to track these on an overall level, but also by node, edge, supplier, month, etc. Hence we want to retain granular history data, so we can filter, slice, and dice as desired.

# Data

The data is regular enough that we can use a relational database schema.

INITIALLY we'll store everything as files, but that's just to avoid the overhead and inconvenience of a proper DB. We'll eventually move, to maybe DuckDB?

A detailed data dictionary is specified in `./scimulator-data-schema.md`

I expect AI to help me to build the data schema based on the information provided above. This will be an interactive process.

# Phases

1. Flesh out design with AI input
2. Build the basic infrastructure
3. Conduct draw-down tests (deterministic demand and supply events; no decision-making)
4. Add supply chain logic (ordering, order fulfillment)
5. Move code to Docker

# Open Questions

* What language will we implement this in?
* How will we incorporate an AI agent to help users manage simulations?
* What will the order and fulfillment logic look like?
* Other?

# Agent Questions (Claude Code Review - 2026-03-21)

## Architecture & Performance

1. **Python performance baseline**: Have you benchmarked a rough target for simulation speed? For example, for a mid-size scenario (say 5,000 customers, 10,000 SKUs, 500 suppliers, 3 distribution layers, 1 year daily resolution), what's an acceptable wall-clock time? This would help determine whether Python+Numba is viable or if we should plan for Rust/Julia from the start.

A: Excluding decision logic, I expect the scenario described to take a few seconds. With decision logic (ordering & fulfillment), this could take 1-2 minutes.

2. **Vectorization strategy**: Are you envisioning a time-stepped simulation (process all events for day 1, then day 2, etc.) or a discrete-event simulation (process events in chronological order regardless of entity)? Time-stepped lends itself much better to vectorization with NumPy/Numba, while discrete-event is more flexible for complex logic but harder to vectorize.

A: I think time-stepped. Note that once a decision is made (ie order more of product X), we can instantly preload future arrivals.

3. **DuckDB timing**: You mention eventually moving to DuckDB. Would it make sense to start with DuckDB from day one? It's file-based (no server overhead), has excellent Python integration, and would give you SQL query capability over simulation results immediately. It could replace Parquet files with minimal friction.

A: Ok, let's do it.

## Data & Input

4. **Synthetic Demand Engine**: You reference this as a source of demand data. Is this an existing tool you've built, or something to be built as part of this project? If existing, where does it live?

A: This exists already in `/scimulator/synthetic_demand_engine`

5. **Geographic model**: You mention ZIP3 as a typical demand aggregation. For transportation cost/time calculations, will we need a distance matrix or rate table as input, or should the simulator compute distances from lat/lon coordinates?

A: We want to support all of the above:
* Haversine distances (with circuity factor)
* Point-point driving distances (e.g. for TLs)
* ZIP3-based zone and speed tables (e.g. what UPS, FedEx use)

6. **Product attributes**: The spec covers supplier and node attributes well, but what product-level attributes matter beyond identity? For example: cube/weight per unit (for capacity constraints), product categories/families (for routing rules), perishability or shelf-life constraints?

A: Yes, we'll want a product master. Key attributes:
* Product ID (SKU)
* Cost (standard unit cost)
* Base Price
* Weight
* Cube
* Orderable Qty

There will also be a product_attribute table, where we can add an arbitrary number of attributes to each SKU, like:
* Brand
* Major product line
* Minor product line
* Etc.

We'll also add more complex relations in the future, like supersessions and kits.

## Simulation Logic

7. **"Bleed down" phase scope**: For Phase 3 (draw-down tests), can you clarify the exact scope? My understanding: pre-loaded inventory at distribution nodes, pre-determined demand arrivals, pre-determined inbound shipments, and the simulator just processes these events and tracks inventory/fulfillment. No ordering decisions, no routing decisions. Is that correct?

A: That is correct. I want to get the basics of the logic right, measure perfornance, design the GUI, etc. with this setup. Adding ordering logic will add a lot of complexity; I just want to get water flowing through the pipes first.

8. **Order splitting**: Can a single customer order be split across multiple fulfillment nodes, or must each order be fulfilled from a single node? This has significant implications for the fulfillment logic.

A: First, we should probably define an "order," which we don't currently. An order is: One event, one customer, one or more SKUs with 1 or more qty each (qty can vary by SKU). Yes, these can be split by SKU (common), or the qty for a SKU can be split (though this is really more of an edge case).


9. **Backorders vs. lost sales**: When demand can't be fulfilled from stock, is demand backordered (queued until stock arrives) or lost? Or is this configurable per scenario?

A. Great question. Let's set a parameter that is a probability of back-order (deferred 'til inventory arrives) vs. lost sale (demand evaporates and is never fulfilled). We may want a more complex model later (e.g. lost sale % increases with expected delivery promise, etc.)

And that's a basic design concern: Basically every component of this model could be made more complicated; as much as possible, we want to modularize to allow that.

## UI & AI

10. **The AI interface section (line 44)** appears to be cut off mid-sentence: "we want to offer the capability for..." — what was the intended completion?

A: We want to offer the capability for the user to instruct an AI agent to make changes or summarize reporting, e.g. "increase demand by 10% for all customers in New England, March-May", or "how much did turns change between scenarios B and C?"

11. **Web framework preference**: Given the simulation complexity and need for real-time progress updates, would you lean toward FastAPI (async support, WebSockets) over Flask?

A: I don't know. Probably.

## Spec Gaps / Minor Items

12. **Line 65**: "Demad Nodes" — typo for "Demand Nodes" - Yes
13. **Line 111**: "multi-model" — should this be "multi-modal"? - Yes
14. **Line 142**: "auper" — typo for "super" - Yes
15. **Line 175**: "Replanishment" — typo for "Replenishment" - Yes
16. **Inventory "states" and transitions**: You mention several states (received, saleable, committed, damaged). Would it be worth defining the allowed state transitions as a state machine? This would make the model more rigorous and help catch edge cases early.

A: I don't know. Please advise on the option here.

## New thoughts

Here are some new thoughts:

* We want to enable data versioning. E.g. if the user instructs the AI to "increase sales for New England customers by 10% in July-August," we want to keep the original dataset as well as the new one. Input data is just another parameter

* We should allow easy import/export of input and output data. NO ACTION HERE YET; this is a ~v2 feature.

* Batching. We want to support the use case of definining many scenarios, which have different parameters and/or input data (as input data is really just a big ball of parameters), and run them all (in batches or all at once). The model will need to be able to support sync job runs. A key part of the UI will be around scenarios. In fact, we'll want powerful views for comparing scenarios, both map-based and table based (e.g. KPIs as rows, scenarios as columns, with diff columns as well). But this is all future UI

Comment: Help me determine a logical order to add all of these "Let's add these in the future" features. That sequence is important. It's also important, I think, to be aware of these as we build even the basic system- drawdown only, one scenario at a time, etc.

# Agent Follow-Up (Claude Code - 2026-03-21, Round 2)

## Responses to Your Answers

### On inventory state machine (Q16)

**Recommendation: Yes, define a lightweight state machine.** Here's why: without explicit transitions, it's easy to accidentally allow impossible moves (e.g. damaged → committed). A simple definition like this is enough for now:

```
in_transit → received → saleable → committed → shipped
                     → damaged → (disposed | repaired → saleable)
saleable → in_transit  (for inter-node transfers)
```

This doesn't need to be a formal library — just an enum of states and a dict of allowed transitions. It costs almost nothing to implement, prevents bugs, and makes the model self-documenting. We can extend it later (e.g. add "quarantine", "returned" states).

A: OK, sounds good!

### On the demand engine integration

I looked at the existing Synthetic Demand Engine. It generates product-level time-series demand (with seasonality, trends, noise, anomalies, correlations, and geographic weights via ZIP3). A few integration questions:

17. **Demand → Orders**: The demand engine produces continuous demand values per product per time step. But the simulator needs discrete *orders* (one customer, multiple SKUs, quantities). The engine has an `order_generator.py` — does it already produce order-level output, or does it only produce aggregate demand curves that we'd need to disaggregate into orders?

A: It produces discrete orders. Only one SKU per "order" as it exists currently. There's a version of the demand engine that didn't work properly; there may be vestiges of that left? I can confirm it can produce proper synthetic demand (or if not we'll fix that in a different thread)

18. **Customer identity**: The demand engine has geographic weights (ZIP3) but no customer concept. For the simulator, do we need named/identified customers (e.g. Customer #1234 at ZIP3 100), or is demand anonymous — just "X units of SKU Y arrive at ZIP3 Z on date D"? This matters for order history tracking and backorder logic.

A: We will want the option of named customers. WHen aggregating to ZIP3 level, the names can just be e.g. "Z017".

### On your new thoughts

19. **Data versioning**: This is a great instinct. Rather than building a custom versioning system, I'd suggest modeling this as **immutable, named datasets** in DuckDB — each input dataset gets a version ID, and scenarios reference dataset versions. This is simple, works with DuckDB from day one, and the AI agent can create new versions via copy-and-modify. Does that match your mental model?

A: Yes. Storage is cheap these days. The only dataset I'm worried about, size-wise, is the daily inventory-by-SKU-by-node-edge table! This should be sparse- a missing value for a day/node/sku/inventory state combo implies inventory = zero.

20. **Scenario model**: You describe scenarios as combinations of parameters + input data. Should we formalize this now? A scenario could be defined as:
    * A reference to an input dataset version (demand, supply schedules, etc.)
    * A set of parameter overrides (backorder probability, node capacities, etc.)
    * Network configuration (which nodes, which edges)
    * A set of policy rules (fulfillment logic, routing rules)

    Even for Phase 3 (drawdown), having this structure in place means we're not refactoring later. Worth defining now, even if most fields are optional?

A: Yes, for sure. Having a robust scenario definition is important from day 0.

## New Questions

### Simulation mechanics

21. **Time resolution switching**: You mention daily with optional hourly. Should a single simulation run mix resolutions (e.g. hourly for outbound fulfillment, daily for inbound replenishment), or is resolution a global setting per run?

A: I think I want a single resolution *per scenario*- e.g. one thing we could change is running the same sim, but more or less granularly. This would be limited by the granularity of the data- if the data is all daily, an hourly sim would have very similar results. But if we run daily with hourly input data, we need to aggregate appropriately. This brings up one issue: A lot of the time, daily will be fine, but we'll want to differentate "early" orders vs "late" ones. This becomes important for e.g. parcel carrier outbound cut-offs. This is a "let's ID this now, but worry about it in the future" issue. For now, assume that the data will match the model time step- hourly resolution data for hourly time steps; daily for daily. This is only for the first phase!!

22. **Warm-up period**: For scenarios with pre-loaded inventory, this isn't an issue. But once we add ordering logic, simulations typically need a warm-up period for inventory to reach realistic levels. Is this something to plan for in the data model (e.g. a "warm-up start" vs. "measurement start" date)?

A: Yes! Good call-out. A really nice feature would be the ability to pre-compute reasonable starting inventory values so we start near-steady-state, but that's like a v12 feature- I have no idea how to do that.

23. **Node capacity enforcement**: When a node exceeds capacity (storage cube, inbound/outbound throughput), what happens? Options: (a) hard constraint — excess is rejected/diverted, (b) soft constraint — overages are allowed but incur penalty costs, (c) configurable. This affects the simulation engine fundamentally.

A: Let's discuss the trade-offs. Option (b) would be the most realistic, generally, until we way overload capacities. I think ordering and replan logic could take (future expected) capacities into account- there could be a lot of complexity here! Please tell me what the fundamental issues are here.

### Architecture decisions

24. **Simulation engine isolation**: For batching/parallelism, the simulation engine should be stateless — all state lives in the DuckDB database per run. This means the engine takes (scenario_config, db_path) and writes results. Are you aligned with this pattern? It makes parallelism trivial (each run gets its own DB file or schema).

A: Yes

25. **Event log vs. snapshots**: For simulation output, do you want: (a) an event log (every shipment, receipt, fulfillment logged as a row — granular but large), (b) daily state snapshots (inventory levels, KPIs per node per day — compact), or (c) both? My recommendation is both — event log for drill-down and debugging, snapshots for fast KPI queries. The event log can be append-only in DuckDB, which is efficient.

A: both. Should be configuerable, though- as I think the state snapshots will be large, we'll want the option of not keeping those. For now, this is a yes/no choice, but we might want e.g. "keep last 30 days" in the future. 

## Recommended Feature Sequencing

Per your request, here's my recommended build order. The principle is: each phase should produce a working, testable system, and each phase should not require rework of prior phases.

**Phase 1: Foundation** (current focus)
1. DuckDB schema + data model (nodes, edges, products, inventory states)
2. Configuration/scenario model (even if mostly unused in drawdown)
3. Demand engine integration (load demand into DuckDB)
4. Basic simulation loop (time-stepped, drawdown only)
5. Event log + snapshot output to DuckDB
6. Basic CLI to run a scenario and dump results

**Phase 2: Visibility**
7. Web UI scaffold (FastAPI + basic frontend)
8. Results dashboard (KPI tables, inventory charts)
9. Network visualization (adapt existing flow_viz)
10. Import/export (upload input data, download results)

**Phase 3: Intelligence**
11. Order fulfillment logic (routing, node selection)
12. Supplier ordering logic (reorder points, forecasting)
13. Transportation mode selection
14. Backorder/lost-sale handling (with the probability parameter you described)

**Phase 4: Scale**
15. Scenario management (save, compare, diff)
16. Data versioning (immutable datasets)
17. Batch runs (parallel scenario execution)
18. Scenario comparison views

**Phase 5: Polish**
19. AI agent interface (natural language → config changes, report queries)
20. Mobile UI
21. Docker deployment
22. Advanced features (cross-docking, rebalancing, supersessions/kits)

**Key architectural decisions to make now** (even if features come later):
- Scenario/config schema — so we don't refactor the data model
- Event log format — so all phases write compatible output
- Plugin/policy pattern — so fulfillment/ordering logic is swappable modules, not hardcoded

Does this sequencing make sense? Anything you'd reorder?

A: It looks good. I initially wanted to swap scale ahead of intelligence, but that probably increases the likelihood of re-work later

# Agent Follow-Up (Claude Code - 2026-03-21, Round 3)

## Capacity Enforcement Deep-Dive (Q23)

You asked for the fundamental trade-offs. Here they are:

**Option A: Hard constraints (reject/divert excess)**
- Pro: Simple to implement — check capacity, reject if exceeded, done.
- Pro: Guarantees physical realism (you can't put 1000 pallets in a 500-pallet warehouse).
- Con: Requires *diversion logic* — where does rejected inventory go? Without a good answer, inventory just vanishes, which breaks conservation. You'd need overflow handling even for the drawdown phase.
- Con: Can cause cascading failures in simulation — one full node causes rejections that overload another node.

**Option B: Soft constraints (allow overage + penalty cost)**
- Pro: Simulation never "breaks" — inventory always has somewhere to go.
- Pro: Realistic for most supply chain situations (warehouses can temporarily overflow into trailers, temporary storage, etc.)
- Pro: Penalty costs naturally surface capacity problems in the KPIs without needing complex diversion logic.
- Con: If penalties are too low, the optimizer (future) will ignore capacity. If too high, it over-corrects. Tuning matters.
- Con: Doesn't model *physical impossibility* — there really is a max for some constraints (e.g. a dock can only process N trucks/day).

**Option C: Configurable per constraint type**
- Pro: Most realistic — storage is usually soft (overflow to temp space), but dock throughput is harder.
- Con: More configuration complexity.

**My recommendation**: Start with **soft constraints only** (Option B). Here's why:
1. In the drawdown phase, there are no ordering decisions, so capacity is mostly observational — you just want to flag "this node went over capacity on these days."
2. Soft constraints require zero diversion logic, which keeps the engine simple.
3. When you add ordering logic (Phase 3), the ordering/fulfillment policies can read capacity utilization and make smarter decisions — this is where capacity actually *matters* for decisions.
4. Later, you can add a configurable "hard cap" multiplier (e.g. 1.5x = allow 50% overage, beyond that reject) for specific constraint types. This gives you Option C without building it now.

The penalty cost structure could be simple: a per-unit-per-day overage cost, defined per node, defaulting to some multiple of the node's variable cost. This shows up in the cost KPIs naturally.

A: I accept your recommendation.

## Sparse Inventory Storage (from Q19 answer)

Your instinct on sparse storage is right. For context on scale: 10,000 SKUs × 20 nodes × 365 days × 5 inventory states = 365M potential rows/year. But if only ~5% of SKU-node combos have non-zero inventory on any given day, that's ~18M rows — very manageable in DuckDB. We should:
- Store only non-zero inventory in snapshots
- Use DuckDB's columnar compression (which handles sparse data well)
- Consider partitioning snapshots by simulation month for fast queries

No question here — just confirming the approach.

A: Confirmed

## Steady-State Initialization (from Q22 answer)

You called this a "v12 feature," but a rough version is actually tractable for Phase 3:
- For the drawdown phase, you manually specify starting inventory, so it's moot.
- For Phase 3 (ordering logic), a simple approach: run the simulation for a "warm-up" period (e.g. 90 days), then discard that period from KPI measurement. The scenario config would have `warm_up_days: 90` and `measurement_start_date`.
- A fancier approach (maybe v4, not v12): compute average daily demand per SKU per node, multiply by target days-of-supply → starting inventory. This is a one-liner per SKU-node and gets you 80% of the way to steady state.

A: Sounds good; let's revisit later

26. **Would you like to include `warm_up_days` in the scenario config from day one?** It's just an integer field that's ignored in the drawdown phase.

A: Yes

## Order Definition Refinement (from Q8/Q17 answers)

You defined an order as: one event, one customer, one or more SKUs with varying quantities, splittable by SKU or (rarely) by quantity. Combined with the demand engine producing single-SKU orders today, I want to confirm the integration path:

27. **For Phase 1 (drawdown), should we treat each demand engine row as a single-line order?** Multi-SKU orders would be a later enhancement where we bundle single-SKU demand rows into orders based on same customer + same day? This keeps the demand engine integration simple for now.

A: In the future, we'll add an order ID to make each order explicit

## Remaining Minor Items

28. **Units of measure**: The spec mentions cube in m3 or cuft, weight but no unit. Should we standardize on metric (m3, kg) or imperial (cuft, lbs), or carry UOM as a field? My recommendation: store in metric internally, display in user-preferred units.

A: Yes, let's standardize on metric, but always support reporting in Imperial. Our base mass unit can be kg; let's use liters for cube. We don't use lengths for products, but may add that; default is cm. Let's use km for distances.

29. **Currency**: Costs are in "$" — is this always USD, or do we need multi-currency support (relevant if simulating networks with Asian/European suppliers)? Recommendation: single currency for now, add exchange-rate tables later if needed.

A: Multi-currency support (but only one currency per scenario- we may break that later, but that covers 99% of cases)

30. **The data schema file** (`scimulator-data-schema.md`) is currently empty (just headers). Once you're happy with this Q&A, I'd like to populate that as one of the first deliverables — it would make everything concrete. Agreed?

A: Yes, I'd love that.

## On Sequencing (your response)

Your instinct to consider Scale before Intelligence is understandable — batch runs are appealing early. But you're right that it increases rework risk. The ordering/fulfillment logic will likely change the event log schema and KPI calculations, and you don't want to build scenario comparison on top of output formats that are about to change. Intelligence → Scale is the safer order.

That said, one small reorder I'd suggest: move **Scenario management** (item 15) into late Phase 2 or early Phase 3. You'll want to save and reload scenarios as soon as you're experimenting with fulfillment logic. The batch/comparison features can stay in Phase 4.

A: OK

---

I think we're converging. After your answers to 26-30, I believe we'll have enough to produce a clean spec and start on the data schema. Let me know!

A: Let's build a clean spec!