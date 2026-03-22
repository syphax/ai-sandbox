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

