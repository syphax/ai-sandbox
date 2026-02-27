# Product Catalog Creator

## Context

We have the ability to create synthetic demand via `scimulator/synthetic_demand_engine`. One input for this engine is a list of products, along with their annual demand/sales in both units and orders.

I want the capability to create a simple artificial product catalog, with the following inputs:

* Name of the dataset
* Number of products
* Length of product name (number of characters)
* Total number of units (across all products)
* Noise factor- used to add some variability to the demand curve
* Shape of demand curve (e.g. log-logistic or lognormal CDF)
* Shape parameters (e.g. k and x0 for log-logistic; mu and sigma for lognormal)
* Quantity:Order mode. Options:
    * Constant ratio
    * Poisson (qty per order follows a Poisson distribution)
* Quantity:Order parameter(s)

The output is a file, products-{name}.csv, with this structure:

part_number,annual_units,annual_orders
AAA,5000,5000
AAB,2000,2000
AAC,150,100
AAD,800,600

The part number is alphabetic, with the length specified by the user. It starts at e.g. AA and increments from there.

## Logic

The total unit demand is distributed, as specified by the user, is distributed across the products according to the selected distribution and parameters. Demand is an integer quantity. Demand is first computed as a float and then rounded. Demand values of zero are permissable.

The number of orders (an integer) is computed as the number of units divided by the quantity per order. The quantity per order is computed according to the selected mode. It may either be based on a constant ratio, or a Poisson distribution. If the mode is Poisson, the user specifies the mean number of units per order. The quantity and orders must either both be zero, or both be non-zero, and the number of orders, must not exceed the number of units.

## Formats

The inputs are provided in a YAML file. This file is provided as an argument to the script. The script is called `create_product_catalog.py` and is located in `/scimulator/utilities`. If no config file argument is provided, the script will use a default config file, located in `/scimulator/utilities/config`. The default config file is called `product_catalog_config.yaml`.

The output is a CSV file, with the structure described above. The output file is located in `/scimulator/utilities/output`. The output file is called `products-{name}.csv`, where `{name}` is the name of the dataset specified in the config file.