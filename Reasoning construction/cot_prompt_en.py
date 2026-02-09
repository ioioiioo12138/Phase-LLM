"""
Chain of Thought Prompt Templates for Materials Phase Prediction

Architecture Design Philosophy:
------------------------------
This template system is inspired by RAAR (Retrieval Augmented Agentic Reasoning) framework,
but adapted for single-agent deep reasoning in strongly-coupled scientific problems.

Key Design Principles:
1. **Expert Perspective Injection**: Forces the model to analyze from structured viewpoints:
   - Thermodynamic Analysis: Composition, phase stability, equilibrium predictions
   - Kinetics & Processing Analysis: Time-temperature, diffusion, transformation pathways
   - Synthesis & Integration: Coupling effects between composition and processing

2. **Single-Agent vs Multi-Agent**: Unlike RAAR's multi-agent debate system, we use a
   single model with multiple perspectives because:
   - Material properties are highly coupled (can't separate composition from processing)
   - Reduces token overhead while maintaining analytical rigor
   - Better for context-intensive scientific reasoning

3. **Search Strategies** (inspired by RAAR's cross-agent strategies):
   - Backtracking: Revisit and correct earlier reasoning points
   - Exploring New Path: Discover overlooked mechanisms or phases
   - Deep Verification: Rigorously validate thermodynamic and kinetic claims
   - Precision Correction: Fix specific errors in phase analysis

4. **Teacher Forcing with Labels**: When model fails repeatedly, provide correct answer
   but require it to construct authentic scientific reasoning path (not just memorization).

Output Format:
-------------
- JSON structure with "CoT" array containing:
  * Multiple "Inner Thinking" steps (with expert perspective titles)
  * "Final Conclusion" (one of three exact phrases)
  * "Verification" (checks completeness and correctness)
  
Final Answer Options:
- "Only FCC and L12 phases form"
- "Other phases form in addition to FCC and L12"
- "L12 phase does not form"
"""

verify_prompt = """<Model Response>  
{}  
</Model Response>  

<Reference Answer>  
{}
</Reference Answer>  

You are verifying a phase formation prediction for alloys based on material composition and reaction conditions.

**Task:**
- Reference Answer is one of three exact phrases:
  * "Only FCC and L12 phases form"
  * "Other phases form in addition to FCC and L12"
  * "L12 phase does not form"
- Extract the prediction from Model Response (look for one of these three exact phrases in the first line after </think> tag)
- If the model used descriptive text instead of exact phrases, map to the correct phrase:
  * Descriptions indicating L12 doesn't form → "L12 phase does not form"
  * Descriptions indicating only FCC and L12 → "Only FCC and L12 phases form"
  * Descriptions indicating additional phases → "Other phases form in addition to FCC and L12"
- Compare extracted prediction with Reference Answer
- Output "True" if they match (same meaning), "False" if they don't match

Output only "True" or "False"."""


query_prompt_init = """<question>
{}
</question>

Please respond to the above question <question> using the Chain of Thought (CoT) reasoning method. This is a materials science question about phase formation based on material composition and reaction conditions. Your response should consist of multiple steps, each of which includes three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

- **'Inner Thinking'**: This is the step where thinking is done. You MUST analyze the problem from multiple expert perspectives in sequence. Each step should start with a brief title indicating the perspective. **Required perspectives** (you may add more if needed):
  1. **Thermodynamic Analysis**: Examine alloy composition (Al, Co, Cr, Fe, Ni, Ti, Nb, etc.), element ratios, phase diagram predictions, and thermodynamic stability of L12 phase versus competing phases (σ, η, Laves, etc.). Consider γ' (L12) formation conditions.
  2. **Kinetics & Processing Analysis**: Analyze heat treatment conditions (homogenization temperature/time, rolling parameters, annealing, aging temperature/time), cooling rates, diffusion kinetics, and precipitation mechanisms. Evaluate whether processing conditions enable or suppress L12 formation.
  3. **Synthesis & Integration**: Integrate thermodynamic and kinetic factors to determine the final phase assemblage. Consider coupling effects between composition and processing.

- **'Final Conclusion'**: At this stage, you summarize the correct reasoning from previous 'Inner Thinking' steps and provide the final answer. The answer MUST be one of these three exact phrases:
  * "Only FCC and L12 phases form"
  * "Other phases form in addition to FCC and L12"
  * "L12 phase does not form"
  No title is required here.
  
- **'Verification'**: At this stage, you verify the conclusion from the "Final Conclusion" step. Check if: (1) all required perspectives were considered, (2) the output format is correct, and (3) the reasoning is consistent. If the conclusion holds, end the process. If not, return to "Inner Thinking" for further reasoning. No title is required here.

The output format must strictly follow the JSON structure below:
```json
{{
"CoT": [
    {{"action": "Inner Thinking", "title": "Thermodynamic Analysis", "content": "..."}},
    {{"action": "Inner Thinking", "title": "Kinetics & Processing Analysis", "content": "..."}},
    {{"action": "Inner Thinking", "title": "Synthesis & Integration", "content": "..."}},
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

# generate initial reasoning process

query_prompt_init_with_decompose = """<question>
{}
</question>
<instruction>
{}
</instruction>

Please answer the above <question> using the **Chain of Thought (CoT)** reasoning method. The <instruction> section provides helpful hints such as decomposition steps or domain knowledge about phase formation that can guide your reasoning process. You may refer to these internally, but **do not explicitly mention them** in your reasoning.

Your response should consist of multiple steps, each of which includes three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

- **'Inner Thinking'**: This is the step where thinking is done. You MUST analyze the problem from multiple expert perspectives in sequence. Each step should start with a brief title indicating the perspective. **Required perspectives** (you may add more if needed):
  1. **Thermodynamic Analysis**: Examine alloy composition, element ratios, phase equilibria, Gibbs free energy considerations, and stability windows for L12 versus competing phases. Consider γ' precipitate formation thermodynamics.
  2. **Kinetics & Processing Analysis**: Analyze processing route (homogenization, deformation, solution treatment, aging), temperature-time profiles, diffusion coefficients, nucleation barriers, and growth kinetics. Evaluate if processing enables phase transformation.
  3. **Synthesis & Integration**: Integrate thermodynamic predictions with kinetic feasibility. Address composition-processing coupling effects and predict final microstructure.

- **'Final Conclusion'**: At this stage, you summarize the correct reasoning from previous 'Inner Thinking' steps and provide the final answer. The answer MUST be one of these three exact phrases:
  * "Only FCC and L12 phases form"
  * "Other phases form in addition to FCC and L12"
  * "L12 phase does not form"
  No title is required here.
  
- **'Verification'**: At this stage, you verify the conclusion from the "Final Conclusion" step. Ensure: (1) all required expert perspectives were addressed, (2) the answer is one of the three valid phrases, and (3) thermodynamic-kinetic coupling was considered. If the conclusion holds, end the process. If not, return to "Inner Thinking" for further reasoning. No title is required here.

The output format must strictly follow the JSON structure below:
```json
{{
"CoT": [
    {{"action": "Inner Thinking", "title": "Thermodynamic Analysis", "content": "..."}},
    {{"action": "Inner Thinking", "title": "Kinetics & Processing Analysis", "content": "..."}},
    {{"action": "Inner Thinking", "title": "Synthesis & Integration", "content": "..."}},
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""



gen_prompt_rethink_Backtracking = """<question>
{}
</question>

<previous reasoning>
{}
<previous reasoning>

<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: You are using **Backtracking Strategy** to revisit earlier reasoning. Break down the analysis using structured expert perspectives:
   - **Thermodynamic Re-analysis**: Re-examine phase stability, composition effects, and equilibrium predictions. Identify where previous thermodynamic assessment went wrong.
   - **Kinetics & Processing Re-analysis**: Re-evaluate processing conditions, time-temperature profiles, and transformation kinetics. Check if kinetic barriers were overlooked.
   - **Corrected Synthesis**: Integrate corrected thermodynamic and kinetic understanding to reach the proper conclusion.
   
   Each step should start with a brief title clarifying its purpose.

2. **Final Conclusion**: Summarize the correct reasoning from all 'Inner Thinking' steps and provide the final answer. The answer MUST be one of these three exact phrases:
   * "Only FCC and L12 phases form"
   * "Other phases form in addition to FCC and L12"
   * "L12 phase does not form"
   No title is needed for this section.
   
3. **Verification**: Verify the accuracy of the "Final Conclusion". Confirm that both thermodynamic and kinetic perspectives were properly revisited. If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion** is incorrect. Your 'Verification' results must align with mine. Proceed to refine the reasoning using **backtracking** to revisit earlier points of reasoning and construct a new Final Conclusion based on materials science principles.

### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "Thermodynamic Re-analysis", "content": "..."}},
    {{"action": "Inner Thinking", "title": "Kinetics & Processing Re-analysis", "content": "..."}},
    {{"action": "Inner Thinking", "title": "Corrected Synthesis", "content": "..."}},
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

gen_prompt_rethink_Exploring_New_Path = """<question>
{}
</question>

<previous reasoning>
{}
<previous reasoning>

<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: You are using **Exploring New Path Strategy** to discover overlooked perspectives. Explore alternative analytical angles:
   - **Alternative Thermodynamic Perspective**: Consider overlooked phases (Laves, σ, η, δ, etc.), metastable equilibria, or composition-dependent phase boundaries that weren't examined before.
   - **Alternative Kinetics & Processing Perspective**: Explore different transformation pathways, non-equilibrium effects (rapid cooling, insufficient aging), or overlooked processing details (cold work, grain size effects).
   - **Novel Synthesis**: Integrate these new perspectives to reach a potentially different conclusion.
   
   Each step should start with a brief title clarifying its purpose.

2. **Final Conclusion**: Summarize the correct reasoning from all 'Inner Thinking' steps and provide the final answer. The answer MUST be one of these three exact phrases:
   * "Only FCC and L12 phases form"
   * "Other phases form in addition to FCC and L12"
   * "L12 phase does not form"
   No title is needed for this section.
   
3. **Verification**: Verify the accuracy of the "Final Conclusion". Confirm that new perspectives were genuinely explored and integrated. If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion** is incorrect. Your 'Verification' results must align with mine. Proceed to refine the reasoning by exploring new approaches to analyzing this alloy system and construct a new Final Conclusion.

### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "Alternative Thermodynamic Perspective", "content": "..."}},
    {{"action": "Inner Thinking", "title": "Alternative Kinetics & Processing Perspective", "content": "..."}},
    {{"action": "Inner Thinking", "title": "Novel Synthesis", "content": "..."}},
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

gen_prompt_rethink_Verification = """<question>
{}
</question>

<previous reasoning>
{}
<previous reasoning>

<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: You are using **Deep Verification Strategy** to validate previous reasoning rigorously. Conduct systematic validation:
   - **Thermodynamic Validation**: Cross-check phase stability calculations, verify composition-dependent phase boundaries, validate equilibrium assumptions. Use phase diagram principles to confirm or refute previous thermodynamic conclusions.
   - **Kinetics & Processing Validation**: Verify time-temperature adequacy, check diffusion distance calculations, validate transformation completion. Ensure processing route supports claimed phase formation.
   - **Integrated Validation & Correction**: Synthesize validation results, identify logical inconsistencies, and construct corrected conclusion with full justification.
   
   Each step should start with a brief title clarifying its purpose.

2. **Final Conclusion**: Summarize the correct reasoning from all 'Inner Thinking' steps and provide the final answer. The answer MUST be one of these three exact phrases:
   * "Only FCC and L12 phases form"
   * "Other phases form in addition to FCC and L12"
   * "L12 phase does not form"
   No title is needed for this section.
   
3. **Verification**: Verify the accuracy of the "Final Conclusion". Confirm that rigorous validation was performed across all perspectives. If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion** is incorrect. Your 'Verification' results must align with mine. Proceed to refine the reasoning by conducting a thorough **validation** process to ensure validity and construct a new Final Conclusion.

### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "Thermodynamic Validation", "content": "..."}},
    {{"action": "Inner Thinking", "title": "Kinetics & Processing Validation", "content": "..."}},
    {{"action": "Inner Thinking", "title": "Integrated Validation & Correction", "content": "..."}},
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

gen_prompt_rethink_Correction = """<question>
{}
</question>

<previous reasoning>
{}
<previous reasoning>

<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: You are using **Precision Correction Strategy** to fix specific errors. Identify and correct mistakes systematically:
   - **Thermodynamic Error Identification & Correction**: Pinpoint specific errors in phase stability assessment, composition analysis, or equilibrium predictions. Provide corrected thermodynamic reasoning with clear justification.
   - **Kinetics & Processing Error Identification & Correction**: Identify mistakes in processing parameter interpretation, transformation kinetics, or time-temperature assessment. Provide corrected kinetic analysis.
   - **Corrected Integration**: Synthesize corrected thermodynamic and kinetic understanding into a unified, error-free conclusion.
   
   Each step should start with a brief title clarifying its purpose.

2. **Final Conclusion**: Summarize the correct reasoning from all 'Inner Thinking' steps and provide the final answer. The answer MUST be one of these three exact phrases:
   * "Only FCC and L12 phases form"
   * "Other phases form in addition to FCC and L12"
   * "L12 phase does not form"
   No title is needed for this section.
   
3. **Verification**: Verify the accuracy of the "Final Conclusion". Confirm that specific errors were identified and properly corrected. If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion** is incorrect. Your 'Verification' results must align with mine. Proceed to refine the reasoning by making precise **corrections** to address prior flaws in understanding phase behavior and construct a new Final Conclusion.

### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "Thermodynamic Error Identification & Correction", "content": "..."}},
    {{"action": "Inner Thinking", "title": "Kinetics & Processing Error Identification & Correction", "content": "..."}},
    {{"action": "Inner Thinking", "title": "Corrected Integration", "content": "..."}},
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

gen_prompt_w_label = """<question>
{}
</question>

<previous reasoning>
{}
</previous reasoning>

<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: You need to construct reasoning that leads to the correct answer through structured expert analysis. Follow these perspectives systematically:
   - **Thermodynamic Deep Dive**: Conduct detailed composition analysis, phase diagram consultation, Gibbs free energy assessment, and stability evaluation for L12 vs. competing phases. Explain thermodynamic driving forces.
   - **Kinetics & Processing Deep Dive**: Perform detailed processing route analysis, time-temperature-transformation assessment, diffusion and precipitation kinetics evaluation. Explain whether processing enables or prevents phase formation.
   - **Scientific Synthesis**: Integrate thermodynamic and kinetic insights to build a coherent, scientifically sound path to the correct conclusion.
   
   Each step should start with a brief title and provide detailed metallurgical justification.

2. **Final Conclusion**: Summarize the correct reasoning from all 'Inner Thinking' steps and provide the final answer. The answer MUST be one of these three exact phrases:
   * "Only FCC and L12 phases form"
   * "Other phases form in addition to FCC and L12"
   * "L12 phase does not form"
   No title is needed for this section.
   
3. **Verification**: Verify the accuracy of the "Final Conclusion". Check if: (1) all expert perspectives were thoroughly addressed, (2) the answer is one of the three valid phrases, (3) the reasoning is scientifically rigorous and educational. If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. 

**Critical Analysis Guidance**: After careful review of experimental evidence and phase diagrams, I need to inform you that the correct answer should be "{}". However, you must NOT simply state this answer. Instead, you need to deeply analyze WHY this is the correct answer through rigorous materials science reasoning:

- If the answer is "L12 phase does not form": Conduct detailed analysis of why L12 thermodynamically or kinetically cannot form (insufficient γ'-forming elements? Temperature outside stability window? Competing phase formation? Inadequate aging treatment?).

- If the answer is "Only FCC and L12 phases form": Provide thorough explanation of why L12 precipitates from FCC matrix (optimal Ti+Al+Nb content? Proper aging parameters? Thermodynamic stability of γ+γ' two-phase field?) AND why competing phases (σ, Laves, η, δ) do NOT form (element levels below threshold? Temperature avoids intermetallic formation?).

- If the answer is "Other phases form in addition to FCC and L12": Identify specific additional phases (σ due to high Cr+Mo? η from Nb? Laves from Ti? TCP phases?) and explain their formation mechanisms (composition exceeds solubility? Temperature favors intermetallic nucleation? Long-term aging promotes undesirable precipitation?).

Your 'Verification' must identify flaws in previous reasoning, then provide NEW 'Inner Thinking' steps with detailed metallurgical justification that naturally leads to this conclusion. Make the reasoning process authentic and educational - as if discovering the answer through scientific analysis, not because it was told to you.

### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "Thermodynamic Deep Dive", "content": "..."}},
    {{"action": "Inner Thinking", "title": "Kinetics & Processing Deep Dive", "content": "..."}},
    {{"action": "Inner Thinking", "title": "Scientific Synthesis", "content": "..."}},
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

reformat_to_complex_cot_prompt = """<Thought Process>
{}
</Thought Process>

<Question>
{}
</Question>

The <Thought Process> above reflects the model's reasoning based on the <Question>. Your task is to rewrite it into a natural thinking process following these requirements:

1. **Format**: The thinking process must be wrapped in <think></think> tags. After </think>, provide:
   - FIRST LINE: The final answer (one of the three exact phrases below)
   - BLANK LINE
   - Brief explanation of WHY this conclusion was reached (be concise and relevant, no unnecessary content)

2. **Final Answer**: Must be one of these three exact phrases:
   - "Only FCC and L12 phases form"
   - "Other phases form in addition to FCC and L12"
   - "L12 phase does not form"

3. **Style**: Use step-by-step natural reasoning with casual transitions like "hmm," "wait," "so," "also."

4. **Content**: Focus on key factors: alloy composition, processing conditions, and their effects on phase formation. Be thorough yet concise.

**Example Output:**
<think>
[natural reasoning steps]
</think>

Only FCC and L12 phases form

This is because the Ti content and aging conditions are optimal for L12 precipitation, while the Cr levels remain below the threshold for σ phase formation.

Return directly the revised response in JSON format as follows:
```json
{{
  "NaturalReasoning": "<think>\n[reasoning steps]\n</think>\n\n[final answer]\n\n[brief explanation]"
}}
```"""

get_final_response_prompt = """<Internal Thinking>
{}
</Internal Thinking>

<Question>
{}
</Question>

The <Internal Thinking> represents your internal thoughts about the <Question>. Based on this, generate a final response following these requirements:

**Requirements:**
1. **Format Structure**:
   - Provide the final answer as the FIRST LINE. The answer MUST be one of these three exact phrases:
     * "Only FCC and L12 phases form"
     * "Other phases form in addition to FCC and L12"
     * "L12 phase does not form"
   - Add a blank line, then provide a brief explanation of WHY this conclusion was reached (be concise and relevant, no unnecessary content)

2. **Content Guidelines**:
   - Provide thorough reasoning steps in natural language
   - Focus on key factors: composition analysis, heat treatment effects, phase stability
   - Be complete and rigorous in analysis

**Example Structure:**
Only FCC and L12 phases form

The Ti content and aging conditions are optimal for L12 precipitation, while other element levels remain below thresholds for competing phase formation.

Output only your response in this format, nothing else."""

# search strategies
search_strategies = [('Backtracking',gen_prompt_rethink_Backtracking),('Exploring New Paths',gen_prompt_rethink_Exploring_New_Path),('Verification',gen_prompt_rethink_Verification),('Correction',gen_prompt_rethink_Correction)]