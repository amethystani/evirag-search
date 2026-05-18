# CDA-7 Annotation Decision Tree

Use this decision tree to classify the cause of disagreement between two conflicting scientific claims.

## Step 1: Is the disagreement about WHEN the research was done?
→ **YES**: Do the claims come from different time periods and does the later claim supersede the earlier?
  - YES → **TEMPORAL**
  - NO (same era, or the earlier claim is not superseded) → continue to Step 2

## Step 2: Is the disagreement about WHAT OUTCOME was measured?
→ Does Claim 1 measure outcome A and Claim 2 measure outcome B, where A ≠ B?
  - Example: "homework improves test scores" vs "homework reduces wellbeing"
  - YES → **OPERATIONAL**
  - NO → continue to Step 3

## Step 3: Is the disagreement about WHO was studied?
→ Different populations, settings, countries, age groups, demographics?
  - Example: "statins benefit high-risk patients" vs "statins have minimal benefit for low-risk patients"
  - YES → **POPULATION**
  - NO → continue to Step 4

## Step 4: Is the disagreement about HOW the study was conducted?
→ Different experimental designs, protocols, measurement tools, or analytical approaches?
  - Example: RCT vs observational study; different control variables; different statistical models
  - YES → **METHODOLOGICAL**
  - NO → continue to Step 5

## Step 5: Is the disagreement about HOW TO INTERPRET the numbers?
→ Same data range but different conclusions about significance, effect size interpretation?
  - Example: "d=0.30 is a meaningful effect" vs "d=0.30 is too small to matter clinically"
  - YES → **STATISTICAL**
  - NO → continue to Step 6

## Step 6: Did Claim 2 ATTEMPT TO REPLICATE Claim 1 and fail?
→ YES → **REPLICATION**
→ NO → continue to Step 7

## Step 7: Is the disagreement about the UNDERLYING THEORY or FRAMEWORK?
→ Competing theoretical models, paradigms, or explanatory frameworks
  - Example: "minimum wage reduces employment (supply-demand)" vs "minimum wage increases employment (demand-side stimulus)"
  - YES → **THEORETICAL**

## Multiple Classes
When disagreement spans multiple causes, list all that apply separated by "+".
Common combinations:
- `METHODOLOGICAL+POPULATION`: Different design AND different population
- `STATISTICAL+METHODOLOGICAL`: Different analysis AND different protocol
- `TEMPORAL+REPLICATION`: Newer study that tried to replicate and failed

## Boundary Cases

### TEMPORAL vs REPLICATION
- **TEMPORAL**: New evidence simply reflects updated knowledge (the old study wasn't "wrong", science evolved)
- **REPLICATION**: New study explicitly attempted to replicate original methodology and got different results

### METHODOLOGICAL vs OPERATIONAL
- **METHODOLOGICAL**: Different *how* of measuring the *same* outcome
- **OPERATIONAL**: Different *definition* of the outcome itself

### POPULATION vs METHODOLOGICAL
- **POPULATION**: Same design applied to genuinely different groups
- **METHODOLOGICAL**: Different recruitment/sampling that accidentally selects different populations
