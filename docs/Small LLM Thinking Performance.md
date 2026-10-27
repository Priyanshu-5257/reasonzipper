You said: 

i have something few things like now all the llm released they think a lot like if you see smaller models like lower than 30b they perform well with thinking but without thinking they struggles even with the simpler task like an 35 line solution of some 8th class maths text book and ask it to identify the line with issue like wrong addition or something. [do not generate a long response keep short, simple and informative] ChatGPT said: You said: okay so my point is suppose i have an extensive thinking model than can we some how reduce the thinking keeping the performance almost same, like i found some models names - thinking capped they released few models that do nearly 40% lessor thinking 

Worked for 7s 

ChatGPT said: You said: there is an also x post from andrej where he mentioned if you see unreadable english in reasoning then conclusion is RL is working ChatGPT said: You said: yes exactly my point is take an model and try to reduce the reasoning token - irrespective of its readable or not [it will like internal reasoning instead of explainable reasoning] ChatGPT said: 

You said: 

i have another idea where we see the token sequence as a point vector, and then we trace it for reasoning tokens and then suppose it takes n_r reasoning tokens/steps then we will just pick the last and first vector difference and ask model to direct drift [i know direct jump will be not possible so will do instead n_r tokens/steps use k_r, where k_r<<n_r 

ChatGPT said: 

You said: 

before can you explain me these 3 paper you just mentioned 

Worked for 46s 

### ChatGPT said: 

You said: 

cool understood, now for glr can we just finetune existing llms or we need to train from scratch ? 

ChatGPT said: 

You said: 

but the point is there any existing support of this kind of heads in the inference engines ? like we can train and all for research but can we use it using vllm, llama.cpp or something 

Worked for 15s 

### ChatGPT said: 

You said: 

no thats where we need to work on i.e we need to find method where we finetune the model with custom head and all so that we 

just finetune and use, and the plus point we have is reasoning tokens need not to be readable 

ChatGPT said: 

You said: 

okay so for the loss objective we can use GRPO like method where we apply KL only to the answer tokens instead of reasoning tokens, and we can also add sft since the target is to keep the answer tokens same and reduce the reasoning tokens. 

ChatGPT said: 

You said: 

okay so now i extended your reasoning capabilities so act like an GOD level research and come up with an methodology 

Worked for 7m 5s 

### ChatGPT said: 

I would build around your GRPO + answer-only KL idea, but add a mechanism that teaches the model how to replace many reasoning steps with fewer ones. 

The central proposal is: 

Distill the computational effect of a long reasoning segment into a few ordinary token IDs. Use custom heads only during training, then discard them. 

Iʼll call this Macro-Step Reasoning Distillation as a working name. This is a proposed methodology —not a demonstrated result. 

One important research update: discrete reasoning tokens are already being explored. DLR learns a discrete vocabulary from compressed reasoning, while Shorthand for Thought learns larger reasoning tokens and evaluates them through vLLM. Therefore, “unreadable tokens compatible with normal inferenceˮ alone would not be a new contribution. Our distinguishing hypothesis should be learning tokens that preserve useful computational progress, rather than reconstructing the original reasoning text. arXiv 1 

# 1. Define the objective correctly 

Let the original model produce: 



We want the fine-tuned model to produce: 



Here, _z_ can contain learned symbols, numbers, ordinary words, or a mixture. Readability is neither rewarded nor penalized. 

The research objective should be: 



For example, we could predeclare at most one percentage point of accuracy loss as the acceptance threshold. That is an experimental requirement, not a promised outcome. 

Importantly, evaluate that constraint on separate difficulty groups—not just the aggregate, which could hide deterioration on harder problems. 

# 2. Keep the deployed architecture ordinary 

The deployed model should remain a standard pretrained Transformer with its usual embedding lookup and next-token prediction head. 

I would initially add 256 learned reasoning tokens, each represented by one token ID. This is a pilot choice, not an assumed optimal vocabulary size. Keep ordinary tokens available too: there is no reason to force an exact numerical value into a tiny abstract codebook. 

|Component|During training|During inference|
|---|---|---|
|Pretrained Transformer, fine-tuned|Yes|Yes|
|Standard embeddings and language-model head,|Yes|Yes|
|including new token rows|||
|Reasoning-segment encoder and quantizer|Yes|No|



|Component|During training|During inference|
|---|---|---|
|Auxiliary checkpoint-prediction heads|Yes|No|
|Frozen teacher model|Yes|No|



Adding vocabulary entries and resizing embeddings is supported in Transformers; expanded reasoning vocabularies have also been evaluated with vLLM. Nevertheless, I would test tokenizer export and generation in both target engines before substantial training, rather than assuming every conversion path handles the checkpoint correctly. Hugging Face 1 

A crucial distinction: we are not algebraically merging an arbitrary custom head into the model. We are using its training signal to change the standard modelʼs weights, then removing that head. 

# 3. Teach compressed reasoning before asking RL to optimize it 

Starting with random symbols and rewarding only the final answer makes the learning problem unnecessarily difficult. 

I would first establish a usable compressed representation. 

## A. Collect verified teacher trajectories 

Use the original reasoning model as a frozen teacher. For each training question, retain a verified answer and its reasoning trace. 

Start with your original task: finding the first incorrect line in a short mathematical solution. Include completely correct solutions, arithmetic mistakes, invalid algebraic transformations, and errors whose consequences propagate into later lines. 

For the initial dataset, use programmatically generated problems where intermediate calculations can also be checked exactly. Later, test on human-written problems and broader mathematical reasoning. 

Split by problem template and operation composition, not merely by random examples. Otherwise, memorizing familiar solution patterns could look like successful reasoning compression. 

## B. Divide reasoning into meaningful blocks 

Instead of treating the entire trajectory as one displacement, identify checkpoints such as: 

An arithmetic operation has been checked; an equation has been transformed; a candidate error has been ruled out. 

For each block, collect teacher representations at its beginning and end, along with any verified intermediate facts. 

Your proposed difference, 



can be an input feature for learning the compression. But I would not make matching that difference the main success criterion: similar vectors do not establish equivalent computation. 

## C. Learn a small discrete representation for each block 

A training-only encoder maps a teacher block into a few discrete codes through a learned codebook. 

Initially, allow several codes per block. Do not require one code to replace an arbitrary amount of reasoning. 

The encoder must only access the teacher prefix up to that checkpoint—not later reasoning or the final-answer section. This reduces a trivial answer-encoding shortcut, although it does not eliminate every form of leakage. 

Most importantly, evaluate the student using codes it generates itself. A model that succeeds only when supplied with teacher-produced codes is a decoder of solutions, not yet a faster solver. 

# 4. Supervise what the codes accomplish—not their English translation 

This is the part I would prioritize as the research contribution. 

After processing a short code sequence, the student should possess information needed for the next useful computation. 

For your error-localization task, auxiliary heads could predict: 

- Whether each checked transition is valid. 

- The correct result of the arithmetic operation being checked. 

- The earliest error found so far. 

These heads read the studentʼs hidden representation before the final answer is supplied. They are discarded after training. 

For broader problems without verified intermediate labels, a weaker alternative is to align selected student checkpoint representations with frozen teacher representations. CODI provides precedent for hidden-state distillation, but representation alignment should remain an auxiliary signal—not 

evidence by itself that reasoning was preserved. arXiv 

The warm-start objective becomes: 



The three terms serve different purposes: 

Code prediction teaches the ordinary language-model head to generate the compressed sequence from the question. 

Answer supervision preserves the desired output. 

Checkpoint supervision encourages the compressed prefix to retain useful intermediate computation. 

During this phase, use hard discrete codes in the forward pass, even when employing a gradient approximation to train the quantizer. Otherwise, the model might learn to rely on soft mixtures of embeddings that will not exist during ordinary inference. 

The important principle is: 

Do not train the codes to reproduce all the teacherʼs prose. Train them to preserve the information needed to continue correctly. 

# 5. Implement your answer-only KL as cross-context distillation 

Your idea is sound, but the conditioning matters. 

The teacher should see its original reasoning _r_ . The student should see its compressed reasoning _z_ . Both should be evaluated against the same canonical answer prefix: 



This asks: 

After shorter reasoning, can the student produce the same answer distribution that the teacher produces after longer reasoning? 

This is valid knowledge distillation despite the different contexts. It is not the same as conventional same-context reference-policy KL. 

I would not feed the studentʼs novel code sequence into the unchanged teacher and assume its answer probabilities are meaningful. The teacher has not learned that code language. 

Also retain answer-only SFT 



During later training, compute this on student-generated prefixes as well as supervised compressed prefixes. 

One subtle but important point: answer-only SFT does not differentiate through the discrete choice of sampled reasoning tokens. That is why code-prediction supervision and policy-gradient learning are still necessary. 

# 6. Use correctness-first GRPO, with controlled length pressure 

After the compressed representation becomes usable, apply your proposed RL stage: 



Keep a small amount of checkpoint-supervised replay during early RL so that the representation does not immediately lose its useful structure. 

There should be no frozen-teacher KL on reasoning tokens. However, retain the usual old-policy ratio/clipping mechanism for stable updates; “no reasoning KLˮ should not mean unrestricted policy changes. 

## Reward design 

For an initial conservative implementation, sample a group of responses for each question. Let _ci_ indicate whether response _i_ is correct and properly formatted, and let ℓ _i_ be its reasoningtoken count. 

Use: 



where _mx_ = 1 only when all sampled responses for that question are correct, and _B_ is a fixed normalization budget. 

This gives a simple curriculum: 

Mixed correct and incorrect responses: improve correctness first. 

All responses correct: prefer the shorter successful trajectories. 

All responses incorrect: do not reward short guesses; rely on supervised replay or better demonstrations. 

This correctness-first gating has precedent in ConciseR, so it should be treated as a strong existing ingredient, not our novelty. arXiv 

## The normalization trap 

Suppose all responses are correct. Their only reward differences come from length. If GRPO divides those rewards by their within-group standard deviation, the scale factor _α_ can largely cancel. 

Consequently, making the length coefficient smaller may not actually weaken the length update. 

For this experiment, I would use group-mean subtraction without group-standard-deviation scaling, and a fixed loss denominator rather than each responseʼs own length. TRL documents both rewardscaling controls and Dr. GRPO-style constant normalization. Hugging Face 

Increase compression pressure only while held-out accuracy remains within the predefined tolerance. Begin around a 2× target reduction, then investigate stronger compression rather than immediately demanding 10. 

Finally, keep the final-answer format fixed. Otherwise, the model could “reduce reasoningˮ by moving its scratch work into the answer section. 

# 7. Run experiments that can reject the hypothesis 

The most important comparison is not against the original verbose model alone. 

|Experiment|What it establishes|
|---|---|
|Original model, normal reasoning|Accuracy and cost reference|
|Answer-only fine-tuning, no reasoning|Whether the task needs intermediate tokens at all|
|Your GRPO + answer-only KL/SFT, existing vocabulary|Whether simple behavioral compression is sufficient|
|Learned codes, without checkpoint supervision|Whether a compressed vocabulary alone helps|
|Full proposed method|Whether computational supervision adds value|



Run these in stages rather than launching a large grid immediately. 

I would add three particularly revealing tests. 

## Teacher-code versus self-generated-code gap 

Compare performance when the student receives teacher-derived codes versus when it generates its own. 

A large gap means the compressed representation is useful but the student cannot reliably produce it. That is a central failure mode, not a minor implementation detail. 

## Matched-length filler-token control 

Train a control using repeated generic thinking tokens with the same number of decoding steps. 

This matters because meaningless filler tokens have supported hidden computation on certain algorithmic tasks. Consequently, unreadable output alone does not demonstrate that a learned codebook carries useful information. arXiv 

If filler tokens perform equally well, we may still have discovered useful reasoning compression— but the specialized vocabulary is unnecessary. 

## Out-of-distribution and deployment tests 

Hold out entire templates, longer solution chains, larger numerical ranges, and unfamiliar combinations of operations. 

Measure accuracy, first-error localization, performance on originally correct teacher examples, reasoning-token distributions, total output tokens, and actual end-to-end latency. 

Benchmark the exported model in stock vLLM and llama.cpp, including after the intended quantization. Count every generated code token, even when the interface hides it. 

Use paired confidence intervals to assess the accuracy difference. “No statistically significant declineˮ is not enough to establish that the model meets a one-percentage-point non-inferiority requirement. 

# 8. The first implementation I would choose 

For a compute-conscious pilot, I would use Qwen31.7B, whose model documentation includes vLLM deployment and local-runtime support, rather than starting with a 27B model. Hugging Face 

Start with a few thousand verified arithmetic/error-localization problems, offline teacher trajectories, LoRA updates, and explicitly trainable new embedding/output rows. Save a fully merged checkpoint with the tokenizer; do not assume an adapter-only export includes everything. 

Most importantly, run your simpler answer-protected GRPO baseline first. It establishes how much compression is available without introducing a new representation. 

Then add discrete codes and checkpoint supervision. A sensible initial acceptance target would be roughly 4050% fewer reasoning tokens, within the chosen accuracy tolerance, and a measured latency improvement. Those are targets to test, not expected results. 

## My research judgment 

The strongest part of your idea is not that reasoning can become unreadable. It is this: 

Can training make a small number of ordinary decoding steps accomplish what previously required a much longer reasoning trajectory, while preserving answer behavior? 

The proposed method gives that hypothesis a mechanism, an engine-compatible implementation, and meaningful failure tests. 

I would pursue computationally supervised compression—not just shorter text, not just hiddenstate similarity, and not gibberish as an objective. 

