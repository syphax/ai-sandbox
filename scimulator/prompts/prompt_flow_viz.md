# Flow Visualization Details

## Design

The aesthetic of this visualization is inspired by the https://earth.nullschool.net/ visualization of wind, waves, and currents. Although it is many years old now, it remains one of the most compelling visualizations ever built. The source code underlying that site is at: https://github.com/cambecc/earth. We also draw inspiration from https://github.com/keplergl/kepler.gl for its modern approach to WebGL-based geospatial visualization.

The Nullschool visualization shows continuous flow fields. Our task is simpler: we are animating discrete flows along fixed origin-destination routes.

## Technology Stack

**Renderer:** WebGL via deck.gl (the same engine underlying Kepler.gl). This supports up to 100K simultaneous particles with GPU-accelerated rendering.

**Frontend:** React + deck.gl + Mapbox GL JS, built with Vite. Standalone page first (no backend), with a clean separation to allow FastAPI server integration later.

**Animation approach:** Keyframe + interpolation. Each flow is stored as a parameterized arc (origin lat/lng, destination lat/lng, start time, end time, attributes). On each animation frame, the client interpolates each active particle's position along its great-circle path. This avoids the memory cost of pre-rendered frames and keeps per-frame computation minimal — modern CPUs and GPUs handle this comfortably at scale.

**Trails:** Implemented using deck.gl's TripsLayer, which renders explicit trail segments with fading. This is the natural fit for deck.gl's full-redraw-per-frame rendering model and is designed exactly for this use case.

**Paths:** Great-circle arcs (not straight lines). This is important for the future addition of inbound flows with overseas container routes across the Atlantic and Pacific.

## Outbound Flows

The intent is to show the geo-temporal distribution of outbound shipments. The interface is a map with a time slider. The user can scrub forward and backward in time, or press play to watch the animation run like a movie.

### Aggregation

The level of aggregation can vary, from individual orders with HH:MM:SS shipment and delivery times, to orders aggregated at the day and postal code (typically ZIP3, but possibly ZIP5) level. Aggregation is used to control the maximum number of on-screen particles; the tentative ceiling is 100K simultaneous particles. In practice, most views will be far below that. When the number of orders "in flight" exceeds 100K, we aggregate by shared origin-destination locations and dates.

### Particle Appearance

Each unit of outbound flow is shown leaving a shipping site on its ship date and arriving at the destination on its delivery date. The unit travels along a great-circle arc at constant speed.

**Size** options:
* Fixed size (as small as one pixel)
* Sized by an order attribute (value, weight, cube, etc.)
* Sized by aggregate volume for flows sharing the same OD attributes

**Color** options:
* None (default white/bright)
* Continuous attribute: value, weight, cube — using Viridis (perceptually uniform, colorblind-safe)
* Delivery days — blue (fast) to red (slow) diverging scale
* Categorical attribute (e.g. brand, product line) — D3 Category10 palette, bucketing into "Other" beyond 10 categories

**Trailing tail:** Each particle can optionally display a fading trail of configurable length (0 to 2 days).

### Origin Sites

Origin warehouses/DCs are shown as small pulsing circles on the map, labeled with their names. In a future iteration, these will become squares or circles encoding pixel-level inventory information (e.g. a 200x200 pixel square representing 40,000 parts).

### Hover/Click Interactivity

When the number of live particles is below 1,000, hovering or clicking a particle shows a tooltip with order details (origin, destination, ship date, delivery date, value, etc.). Above this threshold, tooltips are disabled for performance.

### Filtering (Future — v1)

Users will be able to filter flows by clicking a warehouse (to show only its outbound flows), by destination region, and by other attributes. This is scoped for a later phase.

## Data Structure

The visualization is fully computable from:
* Start and end date/times for the animation window
* A CSV table with columns: origin lat, origin lng, destination lat, destination lng, ship datetime, delivery datetime, and any relevant attributes (value, weight, cube, brand, etc.)
* User-defined parameter settings (see UI section)

Input CSVs may contain hundreds of thousands of rows (e.g. a year of orders), but with average transit times of ~3 days, the number of particles live on-screen at any moment is roughly 1/100th of the total. Client-side processing is viable at this scale.

For the initial version, the CSV file location is set in the global config file and loaded directly by the client.

When the user hits the "Recalculate" button, the client reprocesses the data and rebuilds the animation.

## User Interface

The UI consists of one primary web page. The main element is a full-screen dark-mode map.

### Base Map

Dark basemap with minimal detail: coastlines, country borders, state/province borders, and labeled cities. Origin sites are labeled on the map. Globe projection is aspirational; initial implementation uses a standard web Mercator with dark tiles (e.g. Mapbox Dark or CartoDB Dark Matter).

### Time Controls

A long slider spans the bottom of the map, showing calendar dates. Next to it are play/stop buttons and a speed dropdown (integers 1–5):
* Speed 1 (slow): 1 day = 6 seconds
* Speed 5 (fast): 1 day = 0.5 seconds

The number of speed settings and their mappings are defined in the global config file.

The current date and time is displayed in the upper right corner of the map.

### Sidebar

A collapsible sidebar on the left provides these controls:
* Aggregation level for demand (options: None, Day/ZIP3)
* Attribute for particle size (options: None, Value, Weight, Cube)
* Particle size scaler — slider with text entry, range 1 (single pixel) to 10
* Attribute for particle color (options: None, Value, Weight, Cube, Delivery Days, Categorical)
* Trail length — slider from None to 2 days in 0.5-day increments

A "Recalculate" button appears when any setting is changed.

### Menu

A hamburger menu button (three lines) in the upper left. Initially non-functional; will be populated in future iterations.

## Config File

One config file for initial development. It defines:
* Path to the CSV data file
* Playback speed settings (number of speeds, mapping of speed level to seconds-per-day)
* Default values for all sidebar parameters

Most of these parameters will eventually become user- or organization-level settings.

## Users, Organizations, Security

Phase 1 has no user model; any user-specific data is stored locally in the browser via cookies.

Future phases will support:
* User registration (email, password; email verification)
* User login
* Organization creation
* Organization management: owner, admin, and user roles
