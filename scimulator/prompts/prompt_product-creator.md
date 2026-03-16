# Product Catalog Creator

## Context

We have the ability to create synthetic demand via `scimulator/synthetic_demand_engine`. One input for this engine is a list of products, along with their annual demand/sales in both units and orders.

I want the capability to create a simple artificial product catalog, with the following inputs:

* Name of the dataset: name
* Number of products: cnt_products
* Length of product name (number of characters): length_pn
* Total number of units (across all products): sum_units
* Noise factor- used to add some variability to the demand curve: noise_factor
* Shape of demand curve (e.g. log-logistic or lognormal CDF): demand_curve_shape
* Shape parameters (e.g. k and x0 for log-logistic; mu and sigma for lognormal)
* Quantity:Order mode: qty_per_order_shape. Options:
    * Constant ratio
    * (Any other approaches will be specified in a future iteration)
* Quantity:Order parameter(s) (future feature)
    * Constant ratio: qty_per_order_ratio

The output is a file, products-{name}.csv, with this structure:

part_number,annual_units,annual_orders
AAA,5000,5000
AAB,2000,2000
AAC,150,100
AAD,800,600

The part number is alphabetic, with the length specified by the user. It starts at e.g. AA and increments from there.

## Logic

The total unit demand is distributed, as specified by the user, is distributed across the products according to the selected distribution and parameters. Demand is an integer quantity. Demand is first computed as a float and then rounded. Demand values of zero are permissable.

The number of orders (an integer) is computed as the number of units divided by the quantity per order. The quantity per order is computed according to the selected mode. It may either be based on a constant ratio, or on a method to be defined later. The quantity and orders must either both be zero, or both be non-zero, and the number of orders, must not exceed the number of units.

## Formats

The inputs are provided in a YAML file. This file is provided as an argument to the script. The script is called `create_product_catalog.py` and is located in `/scimulator/utilities`. If no config file argument is provided, the script will use a default config file, located in `/scimulator/utilities/config`. The default config file is called `product_catalog_config.yaml`.

The output is a CSV file, with the structure described above. The output file is located in `/scimulator/utilities/output`. The output file is called `products-{name}.csv`, where `{name}` is the name of the dataset specified in the config file.

# Clarifications:

Demand curve application:

The shape functions take inputs from (0, 1] and return an output from 0-1. To apply to this context, we space the number of parts equally across the range 0-1, then calculate the y value, then scale by total demand, then add noise, and then round.

Note that the log-logistic and lognormal CDF refer to CDF functions; in practice we will need to apply PDF functions.

So concretely for log-logistic: evaluate the log-logistic PDF at x = 1/N, 2/N, ..., 1, then normalize so the sum equals total_units.

# Noise:

The noise should be sampled from a normal distribution of mean 0 and variance of 1. These are then multiplied by a scale factor (which describes the 2-sigma range of the noise) and then multiplied against the computed demand (before rounding). We then round. Example: If the user provides noise_factor = 0.1, then each product's demand is multiplied by (1 + 0.05 * z) where z ~ N(0,1)

I do not think we will need to re-normalize to make the demand totals equal the specified total exactly. We may need to revisit that later.

If the noise results in a negative value, we will set it to zero.

## Order ratios:
For the constant mode, the quantity:order ratio is constant for all parts, subject to rounding.

We will want other modes, but not for v1.

## Part numbers:
If the number of parts exceeds the number of available part numbers, throw an error. Always start with "A"s.

## Other
The script should create the outout folder if it doesn't exist.
The config script should have the option for a random seed.