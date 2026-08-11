# Reading your results

!!! abstract "What this page covers"
    How to read the **sample** page after you click **View** on a row in **All tracked samples**: **Run summary**, **Classification**, **Analysis**, detailed sections, optional **Sample details** / **MNP-Flex**, and **reports**. Content depends on which steps your team enabled.

**Jump to:** [Run summary](#run-summary) · [Classification](#classification-details) · [Analysis](#analysis-details) · [Deep sections](#detailed-results) · [Sample details page](#sample-details-extra) · [MNP-Flex](#mnp-flex) · [Reports](#reports-and-downloads)

For the sample list screenshot, see [All samples](pages-and-routes.md#all-samples) in the tour.

![Sample page: library header and actions, Run summary tiles, Classification details, Analysis details](../images/RunSummary.png)

*Example only: the library ID and metrics in this screenshot are from anonymised quality-control data; your page will reflect your run.*

---

## What you’ll see first: Run summary {#run-summary}

**Run summary** is a block of **tiles** at the top—think of it as a **run card** for the sequencer:

- When the run started and how long it ran  
- Which **device** and **flow cell** were used  
- **Kit**, **basecalling** settings, and other technical metadata  

Use it to confirm you’re looking at the **right run** and **right patient** before you interpret biology below.

Long values (e.g. a full basecall model name) may wrap across the full width so nothing is cut off.

---

## Classification details {#classification-details}

**Classification details** shows how different **classifiers** (for example Sturgeon, NanoDX, PanNanoDX, Random Forest, MARLIN, Lamprey) rank possible tumour classes or methylation-based groups.

**How to read it:**

- Each **card** gives a **main call** and a **confidence** level (often with a coloured bar).  
- **Higher confidence** usually means the model is more certain—but all results still need **clinical interpretation** by your team.  
- **Click a card** to jump to the **expanded section** below with **charts** (bar charts of top classes, and sometimes confidence over time).  
- Reference lines on charts may be labelled **Medium** and **High** to match your lab’s reporting thresholds.

Not every classifier appears for every run; it depends on your pipeline settings.

---

## Analysis details {#analysis-details}

**Analysis details** summarises **coverage**, **copy number (CNV)**, **MGMT methylation**, and **fusion** candidates in a **row of cards**.

**How to use it:**

- Skim the **headline** on each card (depth, CNV status, MGMT level, fusion counts).  
- **Click a card** to scroll to the **longer section** below with plots and tables for that topic.  
- **Coverage** — whether targets are sequenced deeply enough; your lab may use coloured bands or thresholds (e.g. sufficient vs low).  
- **CNV** — gains and losses; plots are for **visual review**—CNV calls are often heuristic.  
- **MGMT** — methylation at the MGMT promoter region; interpretation depends on your clinical protocol.  
- **Fusion** — candidate gene fusions; follow-up may use tables or genome views.

---

## Detailed results (full sections) {#detailed-results}

Below the **Classification details** and **Analysis details** cards, the sample page continues with larger blocks—each tied to the **same topic** as the card above. Your workflow may hide some blocks entirely.

### Classification

The heading **Classification** groups **Sturgeon**, **NanoDX**, **PanNanoDX**, **Random Forest**, **MARLIN**, and **Lamprey (research)** in separate **expandable rows** (click to open). Lamprey is research/evaluation use only.

![Classification section: summary row, expanded Sturgeon with top-classes bar chart and confidence over time](../images/Classification.png)

*Example only: calls and charts are from anonymised quality-control data.*

- **Inside each tool:** a short **summary** (top call, confidence, probe or feature count where shown), then **charts**—typically a **bar chart of top classes** and, where available, **confidence over time** with **Medium** / **High** reference lines.  
- Use this area when you need **more than the dashboard card** shows: full class rankings and how stable the call was as the run progressed.  
- Which tools appear depends on your pipeline configuration.

### Coverage

The **Coverage** block starts with **Coverage Analysis**: an overall **quality** label plus **global** estimated coverage, **targets** estimated coverage, and **enrichment** (how much reads concentrate on targets vs genome-wide).

Further down you will usually find:

- **Per chromosome target coverage** — a chart comparing **on-target** vs **off-target** depth by chromosome.  
- **Coverage over time** — **cumulative** estimated depth (×) as the run advances.  
- **Target coverage over time** — mean target depth over time, with **outlier** highlighting (values beyond about **two standard deviations** from the mean per gene, as described on screen).

Use this section to judge whether sequencing depth is **adequate overall** and whether any time window or gene looks **anomalously low or high**.

### Target coverage

**Target coverage** is the **gene- and region-level** view for your **panel** (for example rCNS2, AML, or a custom panel—whatever your run used). You should see the **panel name**, a note that targets come from the **gene panel BED**, and often an **info** expansion with panel and BED file details.

![Target coverage: per-chromosome distribution with outliers and searchable targets table](../images/TargetCoverage.png)

*Example only: panel and gene labels are from anonymised quality-control data.*

**Gene amplifications:** this section is a good place to notice **suspected amplifications**—genes such as **MYC** or **EGFR** (when on your panel) may show up as **high-coverage outliers** on the per-chromosome plot (often **labelled** on the chart) and/or in the table’s **outlier** column (e.g. flagged relative to mean ± 2 SD). Treat these as **leads** to review alongside the **Copy number (CNV)** section and your lab’s clinical rules; unusually high target depth alone is **not** necessarily proof of amplification.

Typical contents include:

- **Per-chromosome** and **per-gene** views (including scatter, box-style, or bar plots of depth by region).  
- A **searchable table** of targets with coordinates, coverage (×), and outlier flags.  
- **Coverage distribution by gene** and any **threshold** bands your site uses to label sufficient vs low coverage.

Use it when you need **which genes or regions** drove the headline coverage number, not just the genome-wide average.

### MGMT methylation

**MGMT methylation** focuses on the **MGMT promoter** on chromosome 10.

![MGMT methylation: promoter summary, locus plot, latest results, and per-CpG table](../images/MGMT.png)

*Example only: percentages and status are from anonymised quality-control data.*

- **Promoter summary** — **status** (for example methylated vs unmethylated), **average methylation** (%), **prediction score** where shown, and a short note that results come from **per-site** data.  
- **Locus visualization** — a plot of methylation along the promoter region.  
- Often a **table** of individual **CpG** sites with strand-specific coverage.

Interpretation is **protocol-specific**; treat this as supporting information alongside pathology and other assays.

### Copy number (CNV)

The UI uses the heading **Copy number (CNV)** for genome-wide copy-number views.

**Default (genome-wide):** with **Chromosome** and **Gene** set to **All**, you see the full genome on the **CNV scatter** (ploidy) and **difference** (relative) plots, plus sliders to pan and zoom. The **CNV events** table below lists segments (gain/loss, arms or whole chromosomes, affected genes, confidence).

![Copy number (CNV): default genome-wide scatter and difference plots, events summary](../images/CopyNumber.png)

*Example only: events and gene names are from anonymised quality-control data.*

**Plot bin:** the **Plot bin** menu (for example **Data default**, **500 kb**, **1 Mb**, …) **re-bins** the points used for plotting. Choosing a **larger** bin (such as **1 Mb**) often **smooths** noisy regions so **gains and losses** are easier to see at a glance; a **finer** bin can show more detail when you need it. Try a few settings while reviewing the same sample.

![Copy number (CNV): same profile with plot bin set to 1 Mb for clearer smoothing](../images/CopyNumber1Mb.png)

*Example: 1 Mb plot bin for visual inspection; your run may differ.*

**Single chromosome:** set **Chromosome** to one chromosome (for example **chr8**) to fill the plots with that chromosome only—**cytobands** and position on the X-axis make it easier to relate calls to **bands** and genes. Turn **Breakpoints** to **Show** when you want vertical guides at called **breakpoints** on the difference track.

![Copy number (CNV): chromosome 8 only with breakpoints visible](../images/CopyNumberChr8.png)

*Example: focused chr8 view; anonymised QC data.*

**Reading the copy-number axis:** the Y-axis scales to the profile in front of you rather than a fixed range, so a quiet sample gets **fine intervals** (for example 0.1 log2 units) and small gains and losses stay separable. Tick labels are **mirrored on the right-hand edge**, matching the convention of whole-genome methylation (WGM) CNV plots, so a point in the middle of a wide genome-wide panel is never far from a scale to read it against. Minor ticks sit between the labelled ones.

**Colours in the PDF report:** report figures use **blue for gains and red for losses**, and mark **clinical trial target genes in purple** whichever way they went, so a reporting scientist can pick out the altered genes with a trial route at a glance. The convention is stated in the caption under the genome-wide figure, and any figure that actually shows a purple gene carries the note **"Genes labelled in purple represent current clinical trial targets"** in its bottom-right corner — so it travels with the exported PDFs as well as the report. The live view keeps its own palette and is unaffected. The trial gene list ships with a default set (ERBB2, MET, BRCA1/2, MLH1, MSH2, PALB2, RAD51C/D, CDK12, FGFR1/2/3, MTAP) and can be changed per site with `clinical_trial_genes` under `[cnv]` in the workflow TOML; set it to an empty list to turn the highlighting off.

**p/q arm divider:** a faint grey dashed vertical line marks the **centromere** on every CNV plot — in the live view, the report figures and the downloaded PDFs — so an arm-level gain or loss can be read against the arm it affects. The position is the p/q boundary from the packaged GRCh38 cytoband table.

**Reference and segment lines:** a solid line marks **no change** (log2 = 0, or ploidy 2 on the linear scale), with guides at the cut-off and at integer copy numbers. The **Segment line** toggle draws the copy-number level as **flat horizontal bars**, one per segment, the same convention as the methylation-array CNV plots — so the level of a region can be read off directly. Bars are deliberately **detached** rather than joined: a joined line has to rise vertically between levels, and a short unmappable run (a centromere, say) then shows up as a tall spike off the line. A split is only drawn where the change is larger than the bin-to-bin noise, so the line stays flat on a quiet chromosome instead of wandering. Segment lines never bridge a gap in the data or a chromosome boundary.

**Gene labels:** panel genes of interest are marked **on the profile at their own copy-number value**, with the gene name written **rotated** alongside the marker — the same convention as the whole-genome methylation CNV plots. Nothing is reserved for the labels, so the axis keeps its fine intervals however many genes are configured. Target sequencing depth is shown in the marker's tooltip rather than as its height; hover a marker for its copy number, its depth, and how that compares with the panel mean. The **Gene label** menu sets the label size — the default is **Small (4.5 pt)**, with **Tiny (3.5 pt)** through **Largest (8 pt)** available. Point sizes are as rendered in the PDF report, and the live view is scaled to match, so one setting reads the same on screen and on the page.

**Label style — Horizontal or Portrait:** the **Label style** toggle flips gene names between **Horizontal** (read normally, left to right, the default) and **Portrait** (rotated alongside the marker, the methylation-array convention). A horizontal name is as wide as the gene symbol, so names that would overlap are **stacked into lanes** above or below their markers; a rotated name is narrow, so it rarely collides with its neighbours at all. Horizontal is the default because a name that has to be read is easier to read that way — but it uses more of the panel, so switch to Portrait on a densely configured panel where the lanes start to sit over the data.

**Cut-off lines:** the gain and loss thresholds in force are drawn as **dark amber dashed lines** with their value labelled at the end of each line, so gains and losses can be read directly against the cut-off they are judged by. These follow the **Cut-off** menu.

**Coverage genes — All vs Outliers:** **All** marks every configured gene. **Outliers** keeps only genes whose copy number **crosses the gain or loss cut-off** — the same amber lines drawn on the plot. The comparison is against the **genome-wide** baseline, so a gene sitting on a chromosome that is gained end to end (for example **EGFR** on a gained chr7) is reported as gained.

**Genome Y-range — separate from the per-chromosome setting:** the genome-wide view and its **Genome PDF** have their own **Genome Y-range** menu, defaulting to **Auto (fit data)**. That default is deliberate: a fixed window exists so chromosomes can be compared *with each other*, and there is only one genome-wide panel, so fitting it to the data is more useful. Set a fixed span when you want the genome-wide view on the same scale as the per-chromosome plots, or a consistent scale across samples. A chosen span is used **exactly** — it is not rounded out to the next tick, and it is not widened to reach a gene marker; markers beyond it are flagged at the panel edge as they are on the chromosome plots. The **Chr Y-range** menu below is unaffected and continues to govern the single-chromosome views only.

**Chr Y-range — one fixed scale per chromosome:** single-chromosome views, the per-chromosome report plots and the **Chromosome PDFs** download are all drawn on the same fixed Y window, so chromosomes can be compared without re-reading the axis each time. The **Chr Y-range** menu sets it — the default is **±2 log2**, with **±0.6** through **±4** available, plus **Auto (fit data)** to give each chromosome its own axis as before. Ploidy mode uses the equivalent copy-number range. A bin outside the window is not dropped: it is flagged with a small **triangle at the panel edge** at its genomic position, so a deep event such as a homozygous deletion is still visible where it occurs. A **panel gene** whose value is beyond the window is drawn the same way — as an **arrow rather than a dot** — so it is never mistaken for a gene sitting exactly at the axis edge. That matters when comparing with the genome-wide panel, whose axis is fitted to the data and therefore shows such a gene at its true level.

**Per-chromosome plots in the report use each chromosome's full range.** In the PDF report every chromosome gets its own Y range, spanning its deepest and highest bins with no percentile clipping, so a homozygous-depth deletion is drawn at its real value rather than clamped to an edge marker. The purple clinical-trial note is stated **once beneath the block of plots** rather than on each figure — four to a page, it collided with the chromosome titles. The fixed comparison window set by **Chr Y-range** still governs the live view and the **Chromosome PDFs** download, where being able to compare chromosomes against each other is the point; the report is read one chromosome at a time, so depth matters more there.

**NGTD and Clinical Trial Targets CNVs:** a table on the sample page below **Arm / whole-chromosome CNV events**, and in the **Copy Number Variation Detailed Analysis** section of the PDF report listing every configured NGTD / clinical trial target with its result — **GAIN**, **LOSS**, or **No CNVs Detected** — so a reader can see that a target was assessed rather than infer it from an absence. The value shown is the **mean across the gene's bins**, with a focal event kept at its own depth — the same rule the gene markers on the plots use, judged against the cut-off in force. Two further results distinguish cases where nothing was actually assessed, which is not the same as finding no change: **Not on panel** (placed on the genome, but not a target on this sample's panel, so only background coverage is present) and **No data yet** (a panel target that no CNV bin covers yet). A gene ROBIN holds no coordinates for at all is shown as **Not in reference**, and that row says nothing about the locus either way. Overlapping transcripts at one locus are joined with a `/` — **CDKN2B/CDKN2B-AS1**, **MYB/MYB-AS1**, **MYCN/MYCNOS** — because they occupy the same CNV bins and cannot be told apart at this resolution, so they get a single row rather than one that repeats the other. The list is set with `ngtd_genes` under `[cnv]` in the workflow TOML; an empty list removes the table.

**CNV load** appears in three places: as its own card on the sample page under the genome-wide profile, as a titled block in the **Copy Number Variation Detailed Analysis** section of the PDF report immediately above the event tables, and as a one-line header at the top of the first page of the **Genome PDF** and **Chromosome PDFs** downloads. It is reported as a percentage and in Mb, split into gain, loss and total, alongside the span of genome actually assessed.

**CNV load follows the Cut-off menu**, and every place it appears names the cut-off that produced it — for example *CNV load (cut-off ±0.3, calling default)* or *CNV load (cut-off ±0.15, non-default)*. Lowering the cut-off to review a low-purity sample raises the load accordingly, which is the point: change that sits under ±0.3 is exactly what that menu exists to look at. Because of this, **a load figure is only comparable between samples reviewed at the same cut-off** — always read the label with the number. The GUI card, the PDF report and both PDF downloads all carry whatever is set on screen. It counts bins whose log2 ratio crosses the cut-off in force — the same lines drawn on the plots and used for the whole-chromosome, arm and regional event tables — so the number agrees with what you see on the figures. Bins with no data are excluded from both the numerator and the denominator, so patchy coverage reports load over what was assessed rather than diluting it toward zero.

**Whole-chromosome events are decided once.** The **Regional CNV Events** table and the **Arm / Whole-Chromosome Events** table use the same rule for whole-chromosome calls, so they cannot disagree: both arms must exceed the calling threshold, one arm must have over 70% of its bins past it, and both arms at least 40%. Previously the regional table used its own rule — comparing the chromosome against the *spread of all chromosome means* — which became less sensitive the more aberrant the genome was. A chromosome uniformly gained at log2 +0.55 was reported on an otherwise quiet genome and silently dropped once four other chromosomes were aberrant, which is the wrong way round for an aneuploid tumour.

**Every table reads the same track.** The **Regional CNV Events** table is built from the **log2 ratio** (observed ploidy against expected copy number), the same track as the plots, the whole-chromosome and arm tables, the gene states and the CNV load. It previously used the sample-minus-reference difference instead. Because the reference pass re-processes the same reads, the reference inherits the sample's own aberration, and a clear whole-chromosome gain could cancel down to under the cut-off — a sample with an unambiguous chromosome 7 gain reported an empty regional table and *Genes in Gained Regions: 0*. Regions with no uniquely mappable sequence — centromeres, heterochromatin and acrocentric stalks — are reported as **not assessed** rather than as deep losses, and are excluded from the statistics the adaptive thresholds are derived from.

A chromosome that is gained or lost but does not meet the whole-chromosome bar is still reported, as **arm events** for whichever arms qualify — nothing is lost, it is described more precisely.

**How a gene's value is decided, and what it can and cannot resolve:** a gene's value is the **mean of the CNV bins it overlaps**, except that any single bin reaching **three times** the cut-off is reported at its own depth. The mean is what keeps single-bin noise out of the calls; the exception is what stops a focal homozygous deletion being averaged away inside a large gene. Taking the most extreme bin instead — as ROBIN did previously — is a maximum-of-several statistic, so it drifts away from zero the more bins a gene spans: at the noise typical of a real run, a **neutral** gene spanning five bins was called about half the time.

**A limit worth knowing:** at a 50 kb bin width, 8 of the 45 NGTD targets are smaller than one bin, so there is nothing to average — their value is simply that bin's value and carries its full noise. Expect on the order of one spurious gene call per sample from this. It is a resolution limit rather than a rule that can be tightened: widening the window to average over neighbours would fix it, but would also halve the sensitivity to a genuine focal deletion in a small gene such as CDKN2A. Treat a single isolated gene call with no supporting regional or arm event, and nothing visible on the plot, as provisional.

**Which cut-off each table uses:** every CNV output applies the **same** cut-off — the plots, the gene **Outliers** filter, the **Whole Chromosome Events** and **Arm Events** tables, the **Regional CNV Events** table, the **NGTD** table, the gene states and the **CNV load**. That is the configured calling threshold (±0.3 log2 by default) unless the **Cut-off** menu is set to something else, in which case every one of them follows the menu. Each of those outputs names the cut-off it was produced at in its heading, so a number is never shown without the threshold behind it. The regional table additionally sizes its thresholds to the variation in the data, which makes it more conservative on a noisy sample, but a region is only reported when it crosses the cut-off as well. Previously the adaptive rule ran alone and reported regions on a chromosome with nothing on it, because it measured each region against that chromosome's own mean.

**Chromosome PDFs give two pages where they are needed:** in the **Chromosome PDFs** download, any chromosome carrying bins outside the fixed window gets a **second page** headed *full range*, with the axis fitted to the data so those bins are shown at their true value — a homozygous deletion appears at its real depth rather than as an edge marker. Its caption says how many bins were beyond the fixed window. Chromosomes that fit the window get one page as before, so the page count only grows where the extra view earns its place. The main PDF report is unaffected and keeps one plot per chromosome. The genome-wide panel is always fitted to the data — there is only one of it, so there is nothing to compare it against.

**Reading the chromosome layout:** on the genome-wide plot each chromosome is opened by a **solid dark vertical line** and its **number sits just to the right of that line**, at the start of the chromosome rather than in the middle — on a wide panel a centred number is a long way from either edge of the chromosome it belongs to. The **centromere** is a **dashed vertical** inside each chromosome, separating the p and q arms; it is lighter than the boundary so the two are never confused. The same dashed centromere line appears on the single-chromosome views.

**Why the genome-wide and single-chromosome views look different around a gene:** each point on the genome-wide plot is the **mean of several analysis bins**. Averaging tightens the scatter (averaging N bins cuts the spread by √N), which is what makes the genome-wide summary readable for arm- and chromosome-level events. A single-chromosome view is drawn at the analysis bin width, so its cloud is the raw, noisier data and individual bins reach much further from the mean.

**How wide the display bin is:** **at most four analysis bins**, capped at 400 kb, and never coarser than the analysis bins themselves. It is derived from the analysis bin width rather than fixed, because the analysis bin is sized from read depth and is not knowable in advance — across archived samples it ranged from **7 kb to 1.15 Mb**. A single fixed width therefore meant very different amounts of smoothing per run: 400 kb averaged 57 bins on a deep 7 kb run, but did nothing at all on a shallow 431 kb one. On a shallow run where the analysis bins are already coarser than 400 kb, the track is left alone rather than smoothed further.

**Panel gene markers are always taken from the raw analysis bins** in both views, so a focal amplification is never averaged away. That is deliberate — a gene marker should show the true peak — but it means a gene marker and the cloud around it are computed at different resolutions on the genome-wide plot, and the more the cloud is averaged the further it falls short of its own markers. Measured on 384 panel gene markers across eight archived samples, the cloud beside a marker reaches this fraction of the marker's value:

| Analysis bins averaged | 1 | 2 | 4 | 8 | 16 | 36 |
| --- | --- | --- | --- | --- | --- | --- |
| Cloud reaches | 1.00 | 0.95 | 0.92 | 0.79 | 0.68 | 0.65 |

Four is the knee of that curve, which is why it is the limit. Expect the single-chromosome view to still show individual bins above a gene marker — that is raw noise at fine resolution, not a disagreement about the gene's value, which is identical in both.

**Finding a gene marker in a dense profile:** a panel gene is plotted at **its own copy-number value**, which on a finely binned single chromosome means it lands inside a cloud of same-coloured points. Gene markers therefore carry a **white ring**, and their stem a white outline, so both stay traceable where they cross the cloud. The dot itself is kept **small** — a run of panel genes on a gene-dense chromosome reads as a row of fine stems rather than a line of blobs — because it is the ring rather than the size that separates a marker from the bins around it. Without this a gene sitting genuinely above its chromosome's level — **EGFR** on a gained chr7, for example — was plotted correctly but was indistinguishable from the surrounding points, so it looked as though the single-chromosome plot disagreed with the genome-wide one. The values were always the same; only the visibility differed, because the genome-wide view plots fewer, coarser points and so the marker stood out there.

**Stacked gene names:** where several panel genes sit close together — chr17 carries six in the rCNS2 panel — their names are stacked into lanes. Every name in a group steps from the **same starting height**, so each gets its own line, and if the stack would run off the top of the panel the whole group slides down to fit rather than piling up against the edge. If a region is still too crowded, switching **Label style** to **Portrait** makes the names narrow enough that most stop needing to stack at all — that is what Portrait is there for.

**Cut-off menu:** the **Cut-off** menu sets the gain/loss level used for **everything the cut-off drives** — the drawn lines and point colouring, the **Outliers** gene filter, the called regions, the **Whole Chromosome**, **Arm**, **Regional** and **NGTD** tables, the gene states and the **CNV load** — in the live view, in the PDF report and in both PDF downloads. **Calling default** follows the configured calling thresholds; the other entries apply a symmetric log2 cut-off, useful when reviewing a low-purity sample where changes sit below the standard threshold. The selection is per sample and lasts for the session; the standing default is set in the admin panel. Because a change here moves what is *called* and not only what is drawn, every affected table and figure states the cut-off it used — headings read *(cut-off ±0.3, calling default)* or *(cut-off ±0.35, non-default)* — so a report generated at a non-standard cut-off can always be recognised as one.

**The genome-wide figure gets a landscape page:** in the PDF report the genome-wide profile is placed on its **own landscape page** rather than inside the portrait text column. The portrait column is 6.3 in wide, so the figure used to be scaled down to fit and its gene labels shrank with it — a 4.5 pt label printed at roughly 2 pt, which is what made the panel look cluttered. That page is **A4 landscape**, giving the figure **10.5 in** of width against 6.3 in in the portrait column, while keeping the whole report on one paper size. It is placed at its rendered size, so labels print at exactly the size you set in **Gene label** and the 24 chromosomes have room to spread out. Page width is what governs gene label crowding: at a fixed font size, nothing else moves labels apart, and rendering wide then scaling down only shrinks the text.

**The panel fills the canvas.** Space for the axis labels and tick numbers is reserved in fixed inches rather than as a percentage of the figure, so the plot itself reaches the edges. Previously it used only about 77% of the width whatever size the figure was, leaving an empty strip down the right-hand side — and widening the figure widened that strip along with the plot. Now extra width goes to the plot. This also corrects the gene-label spacing calculation, which had been told the panel was as wide as the whole figure and so assumed about 29% more room than it had.

**One page changes orientation, not size.** The report is A4 portrait throughout and rotates to A4 landscape for this single figure, then straight back — so it stays a single-paper-size document. Header, footer, logo and page numbering follow the page. For a wider view than an A4 page can give, use the standalone **Genome PDF** download, which is 24 in across.

The standalone **Genome PDF** download is *not* drawn to page proportions — it has no page to fit, so it uses a wide **24 × 6 in** canvas. Font sizes are absolute points, so widening the canvas buys horizontal room for gene labels without making any text smaller: it is the only lever that reduces label crowding without changing the labels themselves. That makes the download the version to reach for when the panel is dense — it has more than twice the horizontal room the report figure can have on A4. Its height is deliberately *not* scaled up with the width, since crowding on this plot is horizontal.

**Downloading plots:** the **Genome PDF** and **Chromosome PDFs** buttons export the current view as **vector PDFs** — one page for the genome-wide profile, one page per chromosome — using the same figures that appear in the PDF report. The download follows the settings you have on screen: **Y-axis** (linear or log2), **Cut-off**, **Gene label** size, **Chr Y-range**, **Coverage genes** (All / Outliers), **Segment line** show/hide and **Plot bin**.

**Also on screen:**

- **Genome-wide profile** card — **genetic sex**, **bin width**, **variance**, and **gained** / **lost** counts (aligned with the dashboard card).  
- **Controls** — **Colour by** chromosome vs up/down, **linear** or **log** Y-axis, **Cut-off**, **Gene label** size, **Chr Y-range**, **Segment line** show/hide, **Plot bin**, **Breakpoints** show/hide when available.  
- **Main plots** — **CNV scatter** (ploidy) and **difference** (relative deviation) share genomic position; use together with the **events** table.

CNV here is for **rapid visual screening**; it is **not** a replacement for certified copy-number assays or expert review.

### Fusion analysis

**Fusion analysis** summarises candidates from reads with **supplementary alignments** (split mappings suggestive of rearrangements).

- **Candidate summary** — counts for **target panel** vs **genome-wide** pairs and groups.  
- **Target panel** — tables (and often plots) restricted to fusions involving your **assay panel**.  
- **Genome-wide** — broader fusion calls outside the panel, if configured.
- **FusionVis-style model view** — selecting a validated fusion pair displays its parent-gene exon tracks, observed join, breakpoint-local support, and predicted exon-scale fusion products directly above the supporting-read table.

Use the tables to inspect **gene pairs**, support, and grouping; follow your lab’s rules for **confirming** interesting events.

The model is generated from ROBIN's existing fusion alignment rows with the bundled GENCODE annotation; no separate FusionVis installation or service is required. It is an exploratory review view and is not a clinical caller.

---

## Sample details page (extra tools) {#sample-details-extra}

This is a **separate URL** from the main sample dashboard: `/live_data/<library-id>/details`. Open it from **More details** on the main sample page (when shown) or via the [tour → Sample details](pages-and-routes.md#sample-details-tools). The page focuses on **disk paths**, **IGV**, **identifiers**, and **tabular** SNP / fusion / target-gene views—not the large classification and analysis storyboards on `/live_data/<id>`.

### Page header

- **Sample details** heading with **library ID**; **Test ID** appears when ROBIN can read it from the sample’s **identifier manifest** on disk.  
- Intro line listing **IGV**, **sample identifiers**, **SNP** tables, **ITDs / insertions** (when run), **fusion pairs**, and **target genes**.
- **View sample identifiers** — opens a modal with manifest-derived identifier fields (where configured).  
- **Back to sample** — returns to the main sample page (`/live_data/<library-id>`).

### Output location

- **Sample output directory** — full server path to this library’s folder under the ROBIN work directory; status shows **Directory found** or **not found** (with the expected path if missing).

### Analysis center

- On some installs, an **Analysis center** / **Deployment** card shows which ROBIN **center** or deployment label the browser session is using.

### IGV browser

- Embedded **IGV.js** genome browser (**Genome: hg38**): ruler, ideogram, reference sequence, gene annotations, coverage histogram, and read alignments.  
- **target.bam** must exist in the sample output folder for ROBIN to build the interactive viewer; otherwise you see **IGV requires target.bam** and a reminder to run **target analysis** first. When tracks load, ROBIN uses **target-scoped** indexed BAMs—typically **`target.bam`**, or (when present) files such as **`sorted_targets_exceeding.bam`** / **`sorted_targets_exceeding_rerun.bam`** under **`clair3/`**, or **`igv_ready.bam`** under **`igv/`**. **Only alignments from those target-region BAMs are shown**—this is **not** a whole-genome alignment view.  
- **SNP table**, **indel table** (when present), **ITD / insertion events** (when present), and **fusion pairs** table each offer **View in IGV** and/or **row clicks** that **move the browser** to the variant, indel, ITD locus, or fusion breakpoints so you can **inspect pileups** in the loaded BAM. The **Target genes** table does the same for each gene interval.

![IGV browser: hg38 tracks and alignments at a locus opened from the SNP table (View in IGV)](../images/IGVview.png)

*Example only: coordinates, gene, and BAM file name are from anonymised quality-control data.*

### SNP analysis

- Appears when SNP processing has written **`clair3/snpsift_output_display.json`**.  
- **Summary** text may include total variants and counts of **ClinVar significant** variants (germline pathogenic / likely pathogenic, VUS, oncogenic, or somatic tier I/II).  
- **Filters:** **PASS only** (keep rows with `FILTER` = PASS), **ClinVar significant only**, optional **Min QUAL**, and **Reset** to clear filters. A **search** box filters the visible rows. The footer may show how many variants match (e.g. “Showing *n* of *N*”).  
- **ClinVar version:** each table shows which **ClinVar release** was used to annotate the sample and which release is **currently installed** in ROBIN. If a newer ClinVar is installed after annotation, a warning is shown and **Re-annotate with current ClinVar** re-runs **snpEff/SnpSift** on the existing Clair3 outputs (no variant re-calling).  
- The **table** lists columns such as chromosome, position, **REF** / **ALT**, gene, **HGVS.p**, annotation, annotation impact, **CLNSIG** (germline), **ONC** (oncogenic), **SCI** (somatic clinical impact tier), associated disease names (**ONCDN**, **SCIDN**), whether the row is **ClinVar significant**, **FILTER**, **QUAL**, genotype (**GT**), **Details** (expand full fields including germline-only **is_pathogenic** when present), and **View in IGV**.  

![SNP analysis: PASS only and ClinVar significant only enabled, filtered table and summary counts](../images/SNPexample.png)

*Example only: variant shown is from anonymised quality-control data; your counts and rows will differ.*

- **View in IGV** centres the embedded **IGV browser** (above on the page) on that variant for **pileup review**.  
- A separate **indel** table may appear when indel display rows exist, with the same filters and ClinVar columns; **View in IGV** there jumps the browser to the indel locus in the **same target-scoped BAM**.  
- If SNP analysis has not finished or the JSON is missing, you see a short **data not found** message instead of the table.

### Fusion pairs

- Built from processed fusion pickles (**`fusion_candidates_master_processed.pkl`** for the panel first, otherwise **`fusion_candidates_all_processed.pkl`** for genome-wide data).  
- **Table:** fusion pair name, both chromosomes, **breakpoint** coordinates, **supporting reads**, and **View in IGV**. **Click a row** or the action icon to jump the **IGV browser** to both breakpoint neighbourhoods (with padding) so you can **inspect supporting reads** in the **target BAM** (same embedded view as SNPs/indels).  
- Footer text may show **total fusions** and **total supporting reads**. Empty or missing data shows a clear **no pairs** / **run fusion first** style message.

### Target genes

- Data comes from **`target_coverage.csv`** if present, else **`bed_coverage_main.csv`**.  
- **Table:** gene name, chromosome, start, end, **coverage (×)** with colour **badges** by depth band, optional **search**, sortable columns, and **View in IGV**. **Click a row** to open that gene’s interval in IGV (with padding).  
- If neither file exists, this block is omitted.

Sections appear **only when** the underlying files exist and your **workflow** enables the relevant steps.

---

## MNP-Flex (if your site uses it) {#mnp-flex}

**MNP-Flex** is provided commercially by **[Heidelberg Epignostix GmbH](https://epignostix.com/)**. You can only use it under an **agreement with Epignostix**; they supply the **account credentials** ROBIN needs to call their service.

!!! warning "When to run MNP-Flex"

    MNP-Flex is recommended only after at least **12 hours of sequencing data**
    have been generated for the sample. Running it earlier may provide less
    reliable or less representative results.

Some sites show an **MNP-Flex results** block in the sample page. It may include:

- When results were last updated  
- **Classifier** name and version  
- A **hierarchical summary** (class / family / superfamily) and **quality** or **MGMT** side panels  

![MNP-Flex results: toolbar, classifier line, hierarchical summary, QC, and MGMT](../images/MNPFlex.png)

*Example of a successful test output: classifier metadata, hierarchy scores, QC status, and MGMT; your run will show its own values.*

A toolbar may show whether the integration is **idle**, **busy**, or **running**. If you never see this block, your deployment may not use MNP-Flex, or credentials may not be configured.

**Applying your Epignostix credentials in ROBIN:** The ROBIN process that runs the workflow and web UI reads **environment variables** on the **server** (not in the browser). Set the username and password Epignostix gave you before starting ROBIN:

```bash
export MNPFLEX_USERNAME="your-epignostix-username"
export MNPFLEX_PASSWORD="your-epignostix-password"
```

If either variable is missing, ROBIN does not show the MNP-Flex block and will not run the integration.

For the complete operator guide, including secure persistent configuration,
optional API settings, data-transfer considerations, bulk operation, output
files, and troubleshooting, see [MNP-Flex setup](../getting-started/mnpflex.md).

---

## Reports and downloads {#reports-and-downloads}

When analysis is far enough along, you can usually **generate a PDF report** for the sample. The file name is typically based on the **sample ID** and is saved under that sample’s output folder.

Your team may also offer **CSV** or **ZIP** exports of **tabular data**—if enabled, follow the on-screen options and wait for notifications to finish before closing the tab.

If a download fails, check the **notification** area and ask your administrator to confirm disk space and permissions.

---

## Important reminder

ROBIN is for **research use** and **support** to clinical decision-making. **Classification**, **CNV**, and **fusion** outputs are **not** a substitute for full pathological and molecular review by qualified staff.

## See also

- [Tour of the screens](pages-and-routes.md)  
- [Troubleshooting](troubleshooting.md)  
