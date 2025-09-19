# VLFeedback-ET study

## Project Workflow

```mermaid
flowchart TD
  subgraph RAW["Raw Data"]
    RAW_ET[("Raw ET data")]
  end

  subgraph PRE["Preprocessing (ET)"]
    P1["Tobiipy (Raw data -> Fixations)"]
    P2["Assign fixations to Q/A\nGenerate reading metrics"]
    P3["Assign fixations to images\nGenerate image saliency"]
  end

  FIX[("Fixations data")]
  RM[("Reading metrics data (text)")]
  IS[("Image saliency data")]
  RMS[("Reading metrics data (text) synthetic")]

  RAW_ET --> P1
  FIX --> P2
  FIX --> P3

  P1 --> FIX
  P2 --> RM
  P3 --> IS

  subgraph COMPRM["Comparisons RM"]
    COMP_RM["Compare reading metrics between responses"]
    SYN_RM["Generate synthetic reading metrics"]
    COMP_SYN_RM["Compare synthetic reading metrics between responses"]
  end

  RM --> COMP_RM
  RMS --> COMP_SYN_RM
  SYN_RM --> RMS
  COMP_SYN_RM --> RESULTS
  COMP_RM --> RESULTS

  subgraph COMPSAL["Comparisons saliency"]
    SYN_SAL["Compute synthetic image saliency"]
    COMP_SAL["Compare real vs synthetic image saliency"]
  end
  ISS[("Saliency data synthetic")]
  IS --> COMP_SAL
  ISS --> COMP_SAL
  SYN_SAL --> ISS
  COMP_SAL --> RESULTS

  subgraph MODEL["Model Attention"]
    LLAVA["Compute LLava attention\n(on same Qs & As)"]
    COMP_ATTSAL["Compare attention with saliency"]
    COMP_ATTRM["Compa attention with RM"]

  end
  ATT[("Attention data")]
  RESULTS[("Results / Evaluation")]

  %% Flows / labeled arrows using the requested s
 

  LLAVA --> ATT
  ATT --> COMP_ATTSAL
  IS --> COMP_ATTSAL
  ATT --> COMP_ATTRM
  RM --> COMP_ATTRM
  COMP_ATTRM --> RESULTS
  COMP_ATTSAL --> RESULTS
```

## Dependencies
- See requirements.txt for full list of dependencies


### data


### process_et_data 
    #TODO ADD CODE FOR PROCESSING ET DATA

### generate_syntethic_data
    #TODO ADD CODE FOR GENERATING SYNTHETIC DATA calling the generative models

### analyse_data
    #TODO ADD CODE FOR ANALYSE DATA comparing both RM and saliency for chosen vs rejected
    
   

### attention




### attention_saliency
    #TODO ADD CODE FOR COMPARE ATTENTION MLLMS WITH HUMAN SALIENCY