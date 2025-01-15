</details>

---

# 3. `DISCUSSION.md`


```markdown
# Discussion: Step 3

Below are the consolidated answers to the Step 3 discussion questions.

---

## 1. Deciding Which Portions of the Network to Train or Keep Frozen

### When to Freeze the Transformer Backbone & Only Train Task-Specific Layers
1. **Limited Compute or Time**  
   - Freezing the large backbone drastically reduces training time.
2. **Well-Aligned Pretraining**  
   - If the pretrained model’s domain closely matches your tasks, its representations might be good enough.
3. **Small Dataset**  
   - Helps reduce overfitting when labeled data is scarce.

### When to Freeze One Head While Training the Other
1. **Stable / Mature Task Head**  
   - If Task A is already performing well, you can freeze it and introduce Task B without degrading Task A.
2. **Incremental or Sequential Learning**  
   - Avoid catastrophic forgetting by keeping the old head fixed while training the new head.

---

## 2. When to Use a Multi-Task Model vs. Separate Models

### Multi-Task Model
1. **Related / Synergistic Tasks**  
   - If tasks share underlying linguistic knowledge, they can benefit from a common backbone.
2. **Data Scarcity**  
   - Abundant data in Task A can help the shared representations, indirectly boosting Task B.
3. **Resource Efficiency**  
   - A single model deployment can be easier to maintain or run.
4. **Maintenance Simplicity**  
   - One model to update or fine-tune for multiple tasks.

### Separate Models
1. **Divergent Tasks**  
   - If tasks are very different or have conflicting representation needs.
2. **Performance Isolation**  
   - Each task can be optimized individually without risking degradation of the other.
3. **Different Hardware Constraints**  
   - If tasks have different latency or memory requirements, separate deployments might be more flexible.

---

## 3. Handling Data Imbalance (Abundant Data for Task A, Limited Data for Task B)

1. **Loss Weighting**  
   - Adjust the relative importance of each task in the total loss function (e.g., increase weight for Task B).
2. **Sampling Strategy**  
   - Over-sample Task B’s mini-batches or alternate batches between tasks to ensure Task B gets enough attention.
3. **Fine-Tuning Approach**  
   - Train the backbone on Task A first, then fine-tune on Task B, optionally freezing parts of the network.
4. **Data Augmentation**  
   - Generate synthetic data, paraphrases, or other augmented samples for Task B to increase coverage.

---

**End of Discussion**
