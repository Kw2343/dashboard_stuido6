# Model Evaluation Metrics

This guide explains the metrics reported by `model-evaluation.py`.

The evaluator recommends 10 products to each user. It then checks whether those
products appear in the user's held-out purchases.

## Simple Example

Suppose a user bought these two products in the test period:

```text
Actual purchases: [Soap, Vitamins]
```

The model recommends:

```text
1. Shampoo
2. Soap
3. Toothpaste
4. Vitamins
5. Tissues
6. Lotion
7. Bandages
8. Deodorant
9. Mouthwash
10. Sunscreen
```

The model has two correct recommendations:

```text
Soap at position 2
Vitamins at position 4
```

## Recall@10

**Question:** How many of the products the user bought did the model find?

```text
Recall@10 = correct recommendations / actual test purchases
```

In the example:

```text
Recall@10 = 2 / 2 = 1.0
```

The model found every purchased product.

- `1.0` means it found all purchases.
- `0.0` means it found none.
- Higher is better.

## Precision@10

**Question:** How many of the 10 recommendations were correct?

```text
Precision@10 = correct recommendations / 10
```

In the example:

```text
Precision@10 = 2 / 10 = 0.2
```

Two of the ten recommendations were purchased.

- `1.0` means all 10 recommendations were correct.
- `0.0` means none were correct.
- Higher is better.

## Hit Rate@10

**Question:** Did the model recommend at least one purchased product?

In the example, the answer is yes:

```text
Hit Rate for this user = 1
```

If none of the recommendations were purchased:

```text
Hit Rate for this user = 0
```

The final Hit Rate is the percentage of evaluated users who received at least
one correct recommendation.

- `1.0` means every user received at least one correct recommendation.
- `0.0` means no user received a correct recommendation.
- Higher is better.

## MRR@10

MRR means **Mean Reciprocal Rank**.

**Question:** How early did the first correct recommendation appear?

The first correct recommendation in the example is at position 2:

```text
Reciprocal Rank = 1 / 2 = 0.5
```

Other examples:

```text
First correct item at position 1: 1 / 1 = 1.0
First correct item at position 5: 1 / 5 = 0.2
No correct item:                 0.0
```

MRR@10 averages this value across users.

- It considers only the first correct recommendation.
- Correct recommendations near the top receive a higher score.
- Higher is better.

## MAP@10

MAP means **Mean Average Precision**.

**Question:** Were all correct recommendations placed near the top?

Unlike MRR, MAP considers every correct recommendation, not only the first one.

In the example:

```text
At position 2: 1 correct item out of 2 recommendations = 1/2
At position 4: 2 correct items out of 4 recommendations = 2/4

Average Precision = (1/2 + 2/4) / 2 = 0.5
```

MAP@10 is the average of this score across users.

- It rewards models that rank all correct items early.
- `1.0` is a perfect ranking.
- Higher is better.

## NDCG@10

NDCG means **Normalized Discounted Cumulative Gain**.

**Question:** Are correct recommendations near the top of the list?

A correct item at position 1 receives more credit than a correct item at
position 10. The score is compared with the best possible ordering.

For example:

```text
Correct items at positions 1 and 2: high NDCG
Correct items at positions 9 and 10: lower NDCG
No correct items:                  NDCG = 0.0
```

- `1.0` means the correct items are in the best possible positions.
- `0.0` means there are no correct items.
- Higher is better.

## Coverage@10

**Question:** How much of the product catalogue does the model recommend?

```text
Coverage@10 =
unique products recommended to all users / products in the catalogue
```

For example:

```text
Catalogue size:               1,000 products
Unique recommended products:     20 products

Coverage@10 = 20 / 1,000 = 0.02
```

This means the model recommends 2% of the catalogue.

A popularity model often has low coverage because many users receive the same
popular products.

- Higher coverage means more products receive exposure.
- Very low coverage means the model repeatedly recommends a small set.
- Higher is usually better, but only when recommendation relevance remains good.

## Intra-List Diversity@10

**Question:** How different are the 10 products from each other?

For example:

```text
List A: 10 nearly identical vitamin products
List B: vitamins, soap, bandages, shampoo, tissues, and other varied products
```

List B should have higher diversity.

The evaluator combines product metadata such as title, category, description,
features, and store. It converts that text into a MiniLM sentence embedding,
compares every pair with cosine similarity, and calculates:

```text
Diversity@10 = 1 - average semantic similarity
```

This detects meaning, not only matching words. For example, "pain relief
tablets" and "analgesic medicine" can be recognized as similar even though
they use different wording.

- `1.0` means the recommended products are very different.
- `0.0` means they are very similar.
- Higher is generally better.

If `products_clean.csv` is unavailable, the evaluator cannot calculate product
similarity and reports diversity as `0.0`. Generated embeddings are cached in
`evaluation_results/cache/` so later evaluations do not repeat this work.

## Popularity Bias@10

**Question:** Does the model mostly recommend the most popular products?

Each product receives a score based on its popularity rank in the training
data:

```text
Most popular product: score near 1.0
Least popular product: score near 0.0
```

The metric averages these scores across all recommendations.

- A value near `1.0` means the model strongly favors bestsellers.
- A value near `0.0` means it recommends mostly less-popular products.
- Lower is not automatically better.

For a popularity model, a high value is expected. For a personalized or
discovery model, comparing this value helps show whether it recommends beyond
the same popular products.

## Reading the Metrics Together

No single metric tells the complete story.

```text
Recall and Hit Rate:       Does the model find purchased products?
Precision:                 How many recommendations are correct?
MRR, MAP, and NDCG:        Are correct products ranked near the top?
Coverage:                  How much of the catalogue receives exposure?
Diversity:                 Are recommendations different from each other?
Popularity Bias:           Does the model mostly recommend bestsellers?
```

A useful model should balance recommendation accuracy with catalogue coverage
and diversity.

## Current Popularity Model Results

The current evaluation uses:

```text
Users evaluated: 687
Recommendations per user: 10
Recommendation list: the same global 8 Popular + 2 Discovery products
Relevant item: any product in the user's held-out test purchases
```

Current results:

```text
Recall@10:          0.0122
Precision@10:       0.0016
Hit Rate@10:        0.0160
MRR@10:             0.0076
MAP@10:             0.0059
NDCG@10:            0.0079
Diversity@10:       0.8335
Popularity Bias@10: 0.9542
Coverage@10:        0.0002
```

### Simple Interpretation

**Recall@10 = 0.0122**

The model found approximately 1.22% of the products purchased in the test
period. It missed almost 99% of held-out purchases.

**Precision@10 = 0.0016**

Approximately 0.16% of recommendation positions were correct. Another way to
read this is that the model produced many recommendations for each successful
match.

**Hit Rate@10 = 0.0160**

Approximately 1.6% of users received at least one correct recommendation. This
is about 11 users out of 687.

**MRR@10 = 0.0076**

Correct recommendations were rare. When a correct recommendation existed, its
position was not consistently near the top.

**MAP@10 = 0.0059**

The model was poor at ranking all relevant purchases near the beginning of the
Top 10 list.

**NDCG@10 = 0.0079**

The recommendation ordering provides very little useful ranking quality. This
is mainly because most users received no correct product at any position.

**Diversity@10 = 0.8335**

The ten products in the global recommendation list are semantically different
from one another. This is calculated using MiniLM embeddings, so differently
worded products with similar meanings are recognized as related.

This does not mean users receive varied personalized lists. Every user receives
the same ten products. Diversity only measures variety inside that one list.

**Popularity Bias@10 = 0.9542**

The recommendations are concentrated very close to the popular end of the
catalogue. This is expected because eight of the ten positions are explicitly
reserved for popular products.

The value is below 1.0 because two positions are reserved for lower-exposure
Discovery products.

**Coverage@10 = 0.0002**

The model exposes only a tiny fraction of the product catalogue. It recommends
the same ten products to all users, so almost all products are never shown.

### Overall Conclusion

The results make sense for a global popularity baseline:

- The list itself contains semantically varied products.
- It strongly favors popular products.
- It gives almost no catalogue exposure beyond its fixed Top 10.
- It performs poorly at predicting an individual user's future purchases.

The model is suitable as a simple homepage or bestseller baseline. It should
not be treated as a strong personalized recommender. Its main value is to give
personalized models such as CBF and NCF a basic result to beat.
