# Flow Visualization Details

## Design 

The aesthetic of this visualization is inspired by the https://earth.nullschool.net/ visualization of wind, waves, and currents. Although it is many years old now, it remains, in my opinion, one of the most compelling visualizations I've ever seen. The source code for this site is at {# TODO: Insert link to source code}

The Nullschool visualization shows flow fields. Our job is a little easier, as we are just flowing flows along fixed routes.

## Outbound flows

The intent is to show geo-temporal distribution of flows. The interface is a map (e.g. of the US or several states) with a slider control for time. The slider allows the user to scroll forward or backward in time. Or, the user can hit a play button to watch the animation play like a movie.

The level of aggregation can vary, from individual orders to orders aggregated at e.g. the day and postal code (typically ZIP3, but possibly ZIP5) level.

Each unit of outbound flow (at whatever aggregation) is shown leaving a shipping site on the day it ships, and arriving on the date of delivery. The unit is shown along a great circle arc from the origin to the destination, at constant speed. The unit is shown as a circle, as small as one pixel. The size of the unit may be:
* A fixed size (as small as one pixel)
* Sized to show some attribute of the unit (value, weight, cube, etc.)
* Sized to show aggregate volume- volume with the same origin-destination (OD) attributes
    * We have a couple options here:
        * We could preserve each unit as a single entity, but scatter them spatially
        * We cold aggregate them into a larger entity (e.g. larger radius circle)
    * I don't know yet whether to choose one option or allow the user to choose; this is an open design question.

We also may want to allow the option of showing each outbound unit with a trailing trail of fixed length; e.g. any pixel that the unit passes through starts bright, but fades to zero after a specified number of simulated time periods (e.g. 6 hours or 1 day.) This is similar to the Nullschool flow curves, but simpler.

We will offer several choices for the color of the dot. Including: order attribute (value, cube, etc.) and delivery days (e.g. faster is blue, slower is red)

## Data Structure and Calculations

We can completely compute the visualization given:
* Start and end date/times
* Table of geocoded origin (lat/lng), destination (lat/lng), ship date/time, delivery date/time, and any relevant attributes for the order (value, weight, cube, etc.)
    * For the very initial version, this will be contained in a CSV; the location of the CSV on the machine running the code is set in the global config file.
* User-defined parameter settings (see the UI section for the list of parameters)

When the user hits the "Recalculate" button (see UI section), the entire visualization can be calculated.

I don't know yet whether this needs to be calculated on the client side or the server side. I would prefer to do so on the client side, if fast enough.
The calculation determines what each "frame" looks like, so that the user can smoothly play the movie, or scan back and forth using the slider.

## User interface

The user interface consists of one primary web page. The UI primarily consists of a map (dark mode by default).

THe main control is a long slider at the bottom of the map. Bext to the slider are play and stop buttons, plus a drop-down for playback speed. This contains integers 1-5; 1 = slow speed (1 day = 6 seconds); 5 = fast (1 day = 0.5 seconds). The number of speed settings, and what they mean in terms of viz speed vs. calendar time, must be defined in the global config file.

On the left, we have a sidebar which can expand or contract using standard conventions for doing so. 
This sidebar lets the user set the following options:
* Aggregation level for demand (initial options: None, day/ZIP3)
* Attribute for outbound unit size (Options: None, Value, Weight, Cube)
* Outbound unit size scaler- slider with text entry. Value ranges from 1 (smallest size is single pixel) to 10
* Attribute for outbound color. Options: None, Value, Weight, Cube, Delivery Days, Categorical Product Attributes (e.g. brand or product line)
* Outbound flow tail length (slider with values None to 2 days, with increments of 0.5 days- this is an initial guess of what will look good)

Recalculate buttom (only active when one of the above attributes is changed)

In the upper left, there's also a 3 line button for menu controls. Initially, this button has no function, but we will add them later.

## Config File

Initially, there will be one server-side config file. This is mainly for initial development; most of the parameters defined in this file will eventually be converted into user or organization-level parametets.

## Users, Organizations, Security

Initially, we will want to keep the user model simple, but will in the future want to offer a full organizations and permissions model.

For phase 1, we will not have a user model; any user-specific data is stored locally on the user's browser with cookies.

we want to support:
* User registration (email, password; email verification)
* User login
* Orgnnization creation
* Organization management: assignment of owner, admin, users



