# Find Amazon FCs

We have a script that can find warehouses from OpenStreetMap data in /Users/bcc/Code/git/ai-sandbox/data_builders/warehouse_finder.py.

That's pretty good, but I also want to build a robust set of tools that can find Amazon facilities. This can require using diverse sources, including but not limited to:

* OpenStreetMap
* Google Maps
* Amazon job postings (for relevant jobs like warehouse workers, etc.)
* News stories (facility openings)
* Government records (leases, purchases, etc.)

I want a script that scans these sources and adds any new sites it finds. It should also be able to identify sites that have closed, and mark those.

The output is a data file, `amazon_locations.csv`, with a structure similar to `/Users/bcc/Code/git/ai-sandbox/data/warehouses.csv`:

* osm_id
* osm_type
* osm_element_id
* latitude
* longitude
* name
* address
* owner
*building_type,last_updated,search_timestamp,area_sq_ft,street_number,street,city,state,postal_code,postal_code_ext,flag_incomplete_address,schema_version,schema_updated,addresses_parsed_at


