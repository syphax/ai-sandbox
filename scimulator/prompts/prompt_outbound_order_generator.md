# Outbound order generator

This capability takes a set of demand (synthetic or actual) and generates a synthetic set of fulfilled orders, with origin, destination, ship date/time, and destination date/time information.

The main purpose of this functionality is to support testing of flow visualization.

This is not a full-flegded capability; the script can be simple and lightweight. In the future, we will obtain fulfilled order data by using actuals, or via full-fledged distribution simulations.

## Usage

This script is run via `python generate-orders-from-demand.py --config config_file.yaml`

If no parameters are given, the script defers to hard-codes paths for the input data and config file.

## Logic

The script converts demand data into fulfilled orders. It does this by:
* Determining which origin point fulfulls the demand
* Assigning a ship date based on the order date and time
    * Default behavior: ship next day
* Calculating delivery date, based on
    * Haversine distance
    * Effective delivery speed (with or without variability)

The logic for assigning origin locations is this:

* Build a grid of all origins (from the origin file) and destinations (from the destination file); calculate the Haversine distances between each O-D pair
    * Origins and destinations are geocoded based on the geo_data file (see the config section for details)
* For each demand event, choose an origin location
    * Use the `facing_fill_param` to determine the probability that the order is fulfilled from the best (closest) FC
    * If not fulfilled from the closest, use the same logic to determine whether the order is fulfulled from the 2nd closest
    * Iterate until the order is assigned, or we get to the farthest depot.

The delivery speed (in days) is determined by these parameters:

```
delivery_speed:
  units: "mph"
  mean: 450
  stdev: 100
```

Here, 'mpd' means "miles per day"
'mean' is the average speed.
'stdev' is the standard deviation of demand speed

For each fulfilled order, we generate a random delivery speed based on these parameters, and calculate the delivery days.

Once we have delivery speed, we assign shipping and delivery date and time.

The default ship mode is 'Next Day'. For these, all orders are shipped the day after the demand event. The time of shipping is determined by the shipping.time parameter, which is a strong in 24 hour HH:MM:SS format.

Delivery time is determined by adding the delivery days to the ship time. The resulting time is trimmed based on the start and end times for the delivery window (these are also given in HH:MM:SS format). If the delivery time is between the end and start times, the delivery is rolled to the start time of the next day. If the start and end times are identical, all deliveries occur at that time each day.

## Paths

For this section, `.` refers to the root directory of this repo.

The default path fhr the config file is `./scimulator/utilities/config`

The default path for data files is `./scimulator/`; the config file paths specify whether the data files are in `/data` or `output`

## Config file

The config file has the following structure:

```
name: "testing"
files:
  demand_file: 'output/demand_test.csv'
  product_master_file: 'data/product_master.csv'
  geo_file: 'data/zip3.csv'
  origin_file: 'data/origin_fcs.csv'
shipping: 
  mode: 'Next Day'
  time: "12:00:00"
facing_fill_param: 0.90
delivery_speed:
  units: "mph"
  mean: 450
  stdev: 100
```

The `geo_file` data must map locations in the `origin_file` and `demand_file` based on e.g. postal codes (ZIP3 or ZIP5 for the U.S.) 
The choice of what fields to use is determined by the first column in `geo_file`- e.g. if the first column is named `zip3`, both the origin and demand files must both have `zip3` fields. If either file does no, the script terminates and throws an error indicating the field name in the `geo_file`, and which file(s) are missing the relevant field.

## Notes

The code should generally be measurement system (imperial, metric) agnostic- there should be no hard-coded assumptions about units; all unit measures must be parameterized. We want to be able to easily switch between unit systems. This is a fundamental rule!