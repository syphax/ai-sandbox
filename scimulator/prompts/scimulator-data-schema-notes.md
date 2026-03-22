# Data Schema Working Notes

## Primary Keys

As discussed, we're going to use text for primary keys.

And, we decided to add `snapshot_interval_days` as a parameter for storing inventory data.

## Units of Measure

BCC: 

OK. Next topic: We decided to use metric units. That's fine. But I'd like to explictly define the UoM in the dataset, so there's no future confusion (e.g. even if a field is `capacity_l`, what if at some point we have to hack things and use cubic meters there? I'd rather have a field that is explicit, e.g. `capacity_uom` = 'm3'. Further, for things like variable costs, I want to support different cost structures entirely, like variable cost per unit, % of value, per litre, etc. Having an explicit uom field supports that. what do you think?

CC: 

...

Option B: UoM reference table + per-field UoM columns
Same as A, but with a uom lookup table that defines valid units and conversion factors. The per-field columns reference it.

Pro: Conversion logic is centralized (the system can auto-convert between L and m3)
Con: Slightly more infrastructure
My recommendation: Option B. The reference table is small (maybe 20 rows) and pays for itself immediately — the engine can validate UoMs on load and auto-convert when needed. And you drop the _l, _kg suffixes from column names since the UoM is explicit.

For cost structures specifically, I'd use the term "basis" rather than UoM, since pct_value isn't really a unit of measure. So: variable_cost + variable_cost_basis.

Want me to update both the schema and spec with this approach?

## Data vs config

In the current version, a lot of info about suppliers, products, nodes, etc. is in the config file. This is cumbersome as we may have many many entries for each of these.

Instead, I want tables for these entities, and I want tables that defines sets of each. E.g. I want to be able to define a set of suppliers that includes suppliers A, B, and C. (In the future, I want the AI assistant to be able to set these up if the user says "please define a set of suppliers that includes all suppliers in North America and Europe that have an "active" tag). The config files will then refer to sets of:
* Products
* Suppliers
* Customers
* Nodes
* Edges

## Questions on Data vs Config

CC:

I've reviewed the current schema, the spec, and the sample scenario YAML. The direction is clear — the entity tables (supplier, supply_node, distribution_node, demand_node, edge, product) already exist in the data schema, so the data itself is already modeled for the database. The gap is: (1) the config YAML currently embeds all this data inline, and (2) there's no "set" mechanism to select subsets for a scenario.

Questions:

1. **Set versioning**: Should sets be tied to `dataset_version`, or are they independent? I could see it either way — sets are really just "which entities to include," which feels more like scenario configuration than input data. But if sets are independent of dataset_version, we need to think about what happens when a set references an entity that doesn't exist in the dataset.

A: I am summarizing our "live" discussion here:

the clean split would be:

| Mechanism | Applies to | Purpose |
|-----------|-----------|---------|
| **Sets** | suppliers, supply_nodes, distribution_nodes, demand_nodes, edges, products | Which entities participate in the scenario |
| **dataset_version** | demand, initial_inventory, inbound_schedule | Which variant of the input data to use |

A scenario would reference both: set IDs for its network topology, and a dataset_version_id for its simulation inputs. The engine would then filter the versioned inputs against the active sets (e.g., drop initial_inventory rows for nodes not in the distribution node set).

One thing this implies: the entity tables themselves (supplier, product, etc.) would not be scoped to a dataset_version. They'd be global — you load all your entities once, and sets select among them.

2. **Static vs dynamic membership**: You mention the future AI assistant use case ("all suppliers in North America with an 'active' tag"). Should we design for both static sets (explicit member lists) and dynamic sets (defined by a filter/rule, evaluated at runtime)? Or start with static only and add dynamic later? Starting with static is simpler, but if we know dynamic is coming, we could design the table structure to accommodate both from the start (e.g., a `set_definition` with a `type` column: 'static' or 'rule', and a `rule_expression` column for the dynamic case).

A: My thinking was that these would be static assignments, even if expressed initially via a "dynamic" high-level description. But, that does raise the question of what happens when you then add another entity that would've fit the initial expression. I think we should start with static, and log dynamic definitions as a future feature.

3. **One set per entity type per scenario, or composable?** Should a scenario reference exactly one product_set, one supplier_set, etc.? Or should sets be composable (e.g., scenario uses the union of product_set_A and product_set_B)? Single-set-per-type is simpler and probably sufficient if sets themselves can be created freely.

A: I was thinking of one set per scenario; if we want the union, we just define a new set as that union. This could change when we add dynamic sets.

4. **Edge sets and node dependencies**: Edges connect specific nodes. If a scenario's node sets exclude a node that an edge references, that edge is invalid. Should edge sets be explicitly curated, or should the engine auto-filter edges based on the included node sets? Auto-filtering seems more practical (include all edges where both endpoints are in the active node sets), with an optional explicit edge set for further restriction.

A: If there are edges that reference missing nodes, those edges can be pruned from the simulation. And if there are nodes with no edges in or out, same. In both cases, we should log a warning of irrelevant edges or stranded nodes.

5. **Simulation input scoping**: `initial_inventory`, `inbound_schedule`, and `demand` are already scoped to `dataset_version`. When a scenario uses node/product sets, should these tables also be filtered by the sets (e.g., ignore initial_inventory rows for products not in the product set)? I'd assume yes — the engine would join against the active sets and silently drop non-matching rows.

A: That's correct.

6. **What remains in the config YAML?** After moving entities to the DB and adding set references, the scenario config would shrink to: scenario metadata (dates, currency, backorder_probability, etc.), set references (product_set_id, supplier_set_id, etc.), and parameter overrides (scenario_param). Is that the right target state? Or should the config YAML go away entirely in favor of the scenario + scenario_param tables?

A: Honestly, the YAML could go away, but I prefer to keep it as a relatively convenient way to specify scenarios and get them into the scenario tables.

7. **Migration path**: The current sample_scenario.yaml serves as both config and data. Should we support a transitional mode where a YAML can still inline entities (for convenience in testing/bootstrapping), with the loader inserting them into the DB tables before simulation? Or do we go straight to DB-only?

A: I think I like the flexibility of allowing YAML inline entries, for speed of implementation for smaller datasets (mostly for testing)

## Follow-up Questions

CC:

8. **Default / "all" sets**: If a scenario doesn't specify a product_set_id (or any set), should the engine default to "use all entities of that type"? This would make sets optional — you only need them when you want to restrict. Nullable set references on the scenario table would express this naturally.

A: I think sets = null will be rare, but let's allow it.

9. **Set table design — typed or generic?** Two approaches:
   - **Typed**: Separate tables per entity type (`product_set` + `product_set_member`, `supplier_set` + `supplier_set_member`, etc.). Cleaner FK constraints, but 10-12 new tables.
   - **Generic**: One `entity_set` table with a `set_type` column ('product', 'supplier', 'supply_node', etc.) and one `entity_set_member` table with an `entity_id` column. Fewer tables, but no FK enforcement on members, and you'd need to validate set_type/entity_id consistency at the application level.
   Which do you prefer?

A: I think multiple tables is better. I don't think this will cause too many headaches. I'd rather a more robust approach here, than having to worry about FK integrity at the application level.

10. **Supplier sets vs supply_node sets**: The entity hierarchy is supplier → supply_node. Should we have both, or just supply_node_set? If we only have supplier_set, we'd derive active supply_nodes from the active suppliers. If we only have supply_node_set, the supplier table becomes purely informational (grouping/defaults). Or both, where supply_node_set is the primary filter and supplier_set is a convenience that expands to all of a supplier's nodes?

A: I think supplier node, though we'd typically compose sets as "include these suppliers". I think we'll want to toggle specific nodes frequently.

11. **Demand node sets**: Your original list mentions "Customers" as a set type. Demand nodes can be auto-created from demand data (the sample YAML has `demand_nodes: []`). If a scenario has a demand_node_set, should demand rows referencing nodes outside the set be silently dropped? That would let you run a scenario for "East Coast customers only" by defining a demand_node_set, without needing a separate dataset_version. Just want to confirm that's the intent.

A: Yes, sets would be used to define sub-sets within a data version

12. **Set metadata**: Should sets have descriptive metadata (name, description, created_by, created_at) for manageability? I'd assume yes, especially for the future AI assistant use case where you'd want to record how/why a set was created.

A: Yes