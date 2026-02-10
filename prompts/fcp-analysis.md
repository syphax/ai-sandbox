# Analyze part counts by OEM

Please generate the following charts:

- Stacked bar of part counts. First grouping (x-axis) is OEM. 2nd grouping (bars) is part_type_clean. Order for part_type_clean is Genuine, OE, OEM, Unspecified
- Same as above, but as a 100% stacked bar chart
- Format these for side-by-side positioning on a Google Slide.

- Stacked bar of part counts. First grouping (x-axis) is OEM. 2nd grouping (bars) is grouped by Status, but ordered by `Shipping Days`, low to high, with None at the top. Secondary sort is Status.
- 100% stacked bar of the same.
- Format these for side-by-side positioning on a Google Slide.

The charts should be shown in-line (assume the script is run as an interactive notebook, though it should also be runnable directly), and saved if FLAG_SAVE_CHARTS = True

## Chart standards

- Use seaborn where practical, fallback is matplotlib.
- Make axis labels (numbers and titles) fairly large
- Always format axis labels with proper number formatting. large values should be X,XX0 format. Values 0-1 should either be 0.00 or 0%; the latter if context indicates that the axis is a percentage. 
- Im stacked bar charts, try to add the data labels (categories) directly on the bars. But if the text is long, fall back to a legend in the right
- "Format these for side-by-side positioning on a Google Slide." means that the chart size and shape should be appropriate for side-by-side positioning on a Google Slide, with room for a slide title and a 1-2 lines of text above or below the charts
- Default color scheme for a stacked bar chart with <=5 groups is light shades of blue. Use e.g. Tableau for stacked bars with >5 categories




