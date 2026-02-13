# Requirements Document: Synthetic Demand Generation (Phase 1.1)

## 1. Project Objective

To generate a granular, timestamped synthetic order ledger by disaggregating annual part-level demand and geographic weights, specifically accounting for varying order frequencies and quantities per part.

## 2. Input Data Schema

The implementation requires three primary inputs:

* **Parts Master (Enhanced):** A table containing:
* `Part_Number`
* `Annual_Units`: Total units demanded per year.
* `Annual_Orders`: Total number of unique orders per year.


* **Geographic Weights:** A mapping of `ZIP3` to `Weight_Percentage` (summing to 1.0).
* **Temporal Constraints:**
* Operational Window: 07:00 to 22:00.
* Distribution Type: Uniform (Initial).

## 3. Functional Requirements

### 3.1 Order Arrival Logic (The "When")

The engine must treat the arrival of an **order** (not a unit) as the primary event.

* **Order Frequency:** For each part, calculate the daily arrival rate: .
* **Sampling:** Use a **Poisson Distribution** with  to determine how many orders occur for that part on any given day.
* **Timestamping:** For each generated order, assign a time sampled from a **Uniform Distribution** between 07:00 and 22:00.

### 3.2 Quantity Logic (The "How Much")

Once an order event is created, the quantity per order must be determined:

* **Quantity Intensity:** Calculate the expected units per order: .
* **Sampling:** Use a **Poisson Distribution** (specifically a **Zero-Truncated Poisson**) with  to assign the `Quantity` for that specific order.
* *Constraint:* The implementation must ensure  to avoid ghost orders.



### 3.3 Geographic Allocation (The "Where")

* **Independence Assumption:** Each order event, regardless of Part Number, is assigned a `ZIP3` independently.
* **Selection:** Perform a **Weighted Random Sample** using the `Geographic Weights` table.

---

## 4. Phase 2: Advanced Modeling Capabilities

The following features are out of scope for the initial implementation but should be considered in the architectural design for future extensibility:

* **Lumpy Demand Handling:** For parts with high volume but very few orders, replace Poisson with a **Negative Binomial distribution** to better model "bursty" or inconsistent demand.
* **Spatio-Temporal Correlation:** Adjusting  based on the interaction of Time and Geography (e.g., higher demand in ZIP 100 during morning hours vs. ZIP 900 in the evening).
* **Product Clustering:** Modeling dependencies where certain Part Numbers are likely to appear in the same order (Market Basket Analysis).
* **Seasonality Profiles:** Moving from a flat 365-day average to weighted monthly or weekly demand curves.

---

## 5. Output Specification (CSV/Parquet)

| Column | Logic |
| --- | --- |
| `order_id` | Unique UUID per row. |
| `timestamp` | Date + Uniformly sampled time (YYYY-MM-DD HH:MM:SS). |
| `part_number` | From Parts Master. |
| `zip3` | Weighted sample from Geo table. |
| `quantity` | Poisson-sampled value (). |

---

## 6. Validation & Guardrails

* **The "Sum-Back" Test:** The total sum of `Quantity` in the output must be reconciled against the `Annual_Units` input.
* **Order Density Test:** The total count of rows for a part must be reconciled against the `Annual_Orders` input.
* **Constraint Check:** Verify that 100% of timestamps fall within the 07:00–22:00 window.

**Would you like me to create a separate "Test Plan" document that outlines the specific SQL or statistical queries an agent should run to verify these success metrics?**

## 7. Implementation

Implement the code for this demand simulator in this repository (~/Code/git/ai-sandbox/scimulator/)