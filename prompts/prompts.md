I want to build a web app that helps organizations build distribution networks. This will have many capabilities. Let's start with one. I'd like to build a database of existing warehouses nationally. I want to start regionally. For starters, I want to be able to define a location and a radius, and search OpenStreetMap (and later other sources) for warehouses within that area. It can be hard to identify warehouses. For starters, I want to find all buildings with a building=warehouse attribute, and also any building that has an area of at least 200,000 square feet. The latter is more difficult, and may involve calculating area from a bunch of nodes defining the building. 

For step 1, let's start by simply searching for buildings with a "warehouse" attribute. I want a Python script that builds AND updates a database (for starters, a parquet file) of warehouses. Key fields include: address, location (lat/lon), name, openstreetmap ID, owner. We will want more fields; this is just a starter list

---

Please structure this repository. There will eventually be a main app. For now, we are building utility scripts that build the underlying datasets. Also, is this dialog (my prompts and your responses) stored in the repo?

---

Great. Next. I want to build a script that generates and displays a map, with a pin for all the warehouses in the warehouses database

--

