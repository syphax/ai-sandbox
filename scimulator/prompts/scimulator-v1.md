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

# Network Entities

The type of distribution network we want to be able to simulate can be defined in terms of:
* Demand Nodes: Customers
* Supply Nodes: Suppliers
* Distribution Nodes: Distribution Centers, Warehouses, etc.
* Transportation Edges: Links between nodes, with associated costs and constraints
* Logic: The decisions that determine how demand should be fulfilled, distribution nodes get replenished, etc. 

## Demad Nodes

Demand nodes are customers, typically aggregated by geography. Most of the time, we will express demand in terms of dates and ZIP3 geographic areas. However, we don't want the model to rely on that level of resolution; we want the option of arbitrary granularity, down to the individual customer and order event.

The key input to the model is demand data. One source of synthetic demand data is Synthetic Demand Engine which 

# Network Decision-Making


# Measures and Outputs

