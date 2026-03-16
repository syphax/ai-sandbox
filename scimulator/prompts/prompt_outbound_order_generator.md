# Outbound order generator

This capability takes a set of demand (synthetic or actual) and generates a synthetic set of fulfilled orders, with origin, destination, ship date/time, and destination date/time information.

The main purpose of this functionality is to support testing of flow visualization.

This is not a full-flegded capability; the script can be simple and lightweight. In the future, we will obtain fulfilled order data by using actuals, or via full-fledged distribution simulations.

## Usage

This script is run via `python generate-orders-from-demand.py --orders orders.csv --config config_file.yaml`

If no parameters are given, the script defers to hard-codes paths for the input data and config file.

## Logic

The script converts demand data into fulfilled orders. It does this by:
* Determining which origin point fulfulls the demand
* Assigning a ship date based on the order date and time
    * Default behavior: ship next day
* Calculating delivery date, based on
    * Haversine distance
    * Effective delivery speed (with or without variability)

## Config file



