# Instructor feedback — project pivot email chain

Email exchange with Prof Min-Yen Kan after the project critique, regarding the pivot from GDSC2 to News Recommendation with Text Embeddings.

## Group 4 to Prof Min

Dear Prof Min,

We are group 4 and thank you for your feedback on our project during the critique. After discussing the concerns you raised about whether our GDSC2 drug-response task was sufficiently a recommender-system problem, our group has decided to pivot to a new direction that is more clearly aligned with recommendation.

Our proposed new project is News Recommendation with Text Embeddings using the MIND-small dataset.

The main research question we would like to investigate is:
Do semantic text embeddings produce better personalized news rankings than classical TF-IDF representations when both use the same user reading history and recommendation procedure?
For each user, we would use their previously clicked articles to construct an interest profile and rank the candidate articles shown in each later impression. We plan to compare:
a popularity baseline,
TF-IDF + cosine similarity,
frozen semantic text embeddings + cosine similarity, and
if time permits, a small two-tower model trained on the impression click labels.
Our primary evaluation metric would be NDCG@5, with MRR as a secondary metric. We also plan to analyze performance across different history lengths and include ablations on how many recent articles are used to construct the user profile.

One reason we felt this was a stronger direction is that the dataset provides genuine user histories, candidate sets, and click outcomes, so the task is directly a personalized ranking problem rather than response prediction framed as recommendation.

We would be grateful for your thoughts on whether this revised direction and hypothesis are suitable for the project, particularly whether the TF-IDF versus semantic-embedding comparison is sufficiently substantive as the core experiment.

Thank you again for your feedback and guidance.
Best regards,
Group 4

## Prof Min-Yen Kan to Group 4

Dear Patchandi, all:

Thanks for your email and sorry for the slow reply. The new project is more aligned, but as you have pointed out not very substantial.  This is because TF.IDF sparse vectors are generally not as effective as encoded dense vectors.

A good recommendation systems project doesn't have to concentrate merely on accuracy, so you can look at other metrics if you find those interesting as well.  If you are comfortable using LLMs as a judge through prompting you can also use a simulated review panel to assess fitness for particular purposes.

I'll be happy to critique your project more if you'd like.  If your team prefers to get a critique on your newer project idea, you can redo the project critique deck and send me and Sahaj the location of the new deck.  We will still use your old deck for your grades but the feedback can be for your new deck.

Sometime in Week 10–12, we will have another chance to critique your projects as an optional (no marks) milestone.  Feel free to use that if you'd like.

--
Min-Yen KAN (Dr) :: Vice Dean (Undergraduate Studies) :: Associate Professor :: National University of Singapore :: NUS School of Computing

## Actionable takeaways

- Prof Kan considers the pivot well-aligned with recommendation, but flagged the core TF-IDF-vs-embeddings comparison as "not very substantial" because dense encoded vectors generally beat TF-IDF.
- Opened the door to non-accuracy metrics and to LLM-as-judge / simulated review panels for fitness-for-purpose evaluation.
- Optional: redo the critique deck for feedback on the new idea (old deck still counts for grades), and an optional no-marks critique session in Weeks 10–12.
