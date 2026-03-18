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

* TODO: GUI

* Also: we want to offer LLMs as an optional interface to the simulation configuration. In addition to editing tables and config files directly, we want to offer the capability for 

# Application Infrastructure

TODO: Initially, we want to run directly on a local machine (assume a Mac or Linux machine with a Chrome browser). Once we have something mostly working, we want to 

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

# Network Decision-Making


# Measures and Outputs

