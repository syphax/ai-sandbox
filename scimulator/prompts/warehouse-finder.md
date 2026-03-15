I want to build a web app that helps organizations build distribution networks. This will have many capabilities. Let's start with one. I'd like to build a database of existing warehouses nationally. I want to start regionally. For starters, I want to be able to define a location and a radius, and search OpenStreetMap (and later other sources) for warehouses within that area. It can be hard to identify warehouses. For starters, I want to find all buildings with a building=warehouse attribute, and also any building that has an area of at least 200,000 square feet. The latter is more difficult, and may involve calculating area from a bunch of nodes defining the building. 

For step 1, let's start by simply searching for buildings with a "warehouse" attribute. I want a Python script that builds AND updates a database (for starters, a parquet file) of warehouses. Key fields include: address, location (lat/lon), name, openstreetmap ID, owner. We will want more fields; this is just a starter list

---

Please structure this repository. There will eventually be a main app. For now, we are building utility scripts that build the underlying datasets. Also, is this dialog (my prompts and your responses) stored in the repo?

---

Great. Next. I want to build a script that generates and displays a map, with a pin for all the warehouses in the warehouses database

--

This has worked great so far. However, I need debugging help. I changed the starter lat/lon cooordinates in warehouse_finder.py, centered right near an Amazon warehouse that is labelled as a warehouse. But, when I run the script, it reports finding zero warehouses. I don't know why it doesn't find any

--

So for the one I was trying to capture, the building type is industrial.

For view_warehouses: I'd like to also add a link in the tooltip, under the link to the OSM map, that takes you to the same location (lat/lon) in Google Maps, as well.

--

So rather than add a boolean, I think I'd like to add a list of building types to include, and a list to exclude. 

Ok, so far we filter based on building type. Next, I want the option to *also* ID and store large buildings (which in many cases are warehouses) by building size. I am thinking that a quick algorithm is to simply take the min/max of the lat/lon of all the nodes for a building, and multiply the differences to estimate maximum area. This doesn't give a precise meaurement of a building's area, but I think it would be a fast way to apply a rough filter for building size? I am open to other options that are fast but perhaps more precise. I'd like to define a min area size, like 100,000 sq ft. I'd like to also be able to define filters for exclusions, e.g. for buildings labeled as office buildings.

--
(This iteration took a long time to run, as it queried all buildings in an area)
--

Let's kill it. Does the query first filter for excluded building types, or does it first query for all buildings? I think it would be better to first filter by the exclusions

--
# Prompt 9: Add config file:
In data_builders/warehouse_finder.py, we currently hard-ocde a lat, lon, and radius. I'd like to add a config file (YAML) where we can add more then one set of lat/lon/radii. And, we should have a parameter called name for each set. The script should read this YAML file and loop through each set of parameters, which are structured as a dict.

The config file should be called warehouse_finger.yaml, and be in a cfg directory which is on the same level as the data_builders directory.