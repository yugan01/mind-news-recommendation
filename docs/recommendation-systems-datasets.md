# Recommendation Systems Datasets
**CP4285 Modern Recommendation Systems — Auxiliary project resource**
CP4285 Instruction Team
15 Sep 2026

Scan for the slides: `soc-n.us/cp4285-t2610-datasets`

# 01 Overview
**CHANGELOG**
* 15 Sep 2026 — Created the project-planning guide, expanded the catalogue to the full source inventory, and added a deck QR link.

# What this deck is for
* Find a public, reproducible dataset that makes a recommendation question possible.
* Match the data modality to the claim: ratings, sessions, text, social links, location, or impressions.
* Start small enough to establish classical and neural baselines before attempting a novel model.

> **Warning**
> Dataset scale is not project quality. A well-scoped question, credible split, error analysis, and ethical critique matter more than a large download.

# How to read a dataset slide
* **RECORDS (Published count):** Interactions, impressions, or check-ins
* **FOOTPRINT (Download / RAM):** Published where available; otherwise an estimate
* **FIT (Task + risk):** The research claim the data can support

* **Published download** is the provider’s compressed archive size when stated.
* **Planning RAM** is an indicative laptop/cluster budget for a parsed working subset; it varies with filtering, features, framework, and batch size.
* Every team must check the current licence, documentation, and terms before download.

# 02 Dataset Catalogue

# Catalogue key
The catalogue is a **source inventory**, not a recommendation that every dataset is equally suitable or still equally accessible.
* ★ has a full project-planning slide in Section 03.
* Entries marked **textbook** were identified in the course books but are not separate RecBole entries.
* Dataset availability, licences, versions, and download sizes change; check the linked provider before building.

# Catalogue: shopping and transaction data I
| Dataset family / version | Primary signal | Detail |
| :--- | :--- | :--- |
| **Amazon Reviews 2014** | Reviews + product metadata | 24 categories; 142.8M reviews |
| **Amazon Reviews 2018** | Reviews + product metadata | 29 categories; 233.1M reviews |
| ★ **Amazon Reviews 2023** | Reviews, metadata, images | 44 categories; 571.5M reviews |
| **Amazon-M2** | Multilingual customer sessions | Six locales |
| **Alibaba-iFashion** | Outfit composition + item attributes | Fashion compatibility |

# Catalogue: shopping and transaction data II
| Dataset family / version | Primary signal | Detail |
| :--- | :--- | :--- |
| ★ **Epinions** | Ratings + directed trust links | Social/trust recommendation |
| **Yelp 2018 / 2020 / 2021 / 2022 / full** | Reviews, businesses, tips, check-ins | Versioned business-review releases |
| ★ **Yelp Open Dataset** | Reviews + business attributes | Current scoped project option |
| **Tmall** | E-commerce interactions | IJCAI-16 competition data |
| **Ta Feng** | Grocery transactions | Nov 2000–Feb 2001 |

# Catalogue: sessions and e-commerce logs
| Dataset | Primary signal | Detail |
| :--- | :--- | :--- |
| ★ **Diginetica** | Search sessions, clicks, purchases | Queries, product metadata, prices |
| ★ **YOOCHOOSE / RecSys Challenge 2015** | Clickstream sessions + purchases | Sequential ranking benchmark |
| ★ **Retailrocket** | Views, carts, transactions | Raw hashed events |
| **Tmall sessions (textbook)** | E-commerce sessions | 1.77M sessions in the course book table |
| **XING (textbook)** | Job-search / interaction logs | Sequential/industrial recommendation |

# Catalogue: advertising and CTR prediction
| Dataset | Primary signal | Detail |
| :--- | :--- | :--- |
| **Criteo** | Ad impressions + click labels | Several days of traffic |
| **Avazu** | Ad click-through logs | CTR prediction contest |
| **iPinYou** | Real-time bidding logs | Three competition seasons |
| **AliEC** | Taobao display-ad clicks | CTR prediction |

# Catalogue: places and spatiotemporal data
| Dataset | Primary signal | Detail |
| :--- | :--- | :--- |
| **Foursquare** | Check-ins, time, GPS, venue semantics | NYC and Tokyo collection |
| ★ **Gowalla** | Check-ins + social graph | 6.44M check-ins |

# Catalogue: movies, video, and streaming
| Dataset | Primary signal | Detail |
| :--- | :--- | :--- |
| ★ **MovieLens 1M** | Explicit movie ratings | Detailed project slide; MovieLens family |
| **MovieLens 100K / 30M (textbook)** | Sessionised rating events | Course-book benchmark variants |
| **Netflix Prize** | Movie ratings | Official competition data |
| **Douban Movie** | Movie comments | Chinese social-media context |
| **Twitch-100K / Twitch-full** | Streams + chat connections | 43-day collection |

# Catalogue: music and audio
| Dataset | Primary signal | Detail |
| :--- | :--- | :--- |
| ★ **Last.fm 2K** | Listening, tags, friendships | HetRec 2011 release |
| **LFM-1b** | Timestamped listening events | 1B+ events; 120K+ users |
| **Yahoo Music** | Music-community preferences | Artist preference snapshot |
| **KGRec-music / KGRec-sound** | Implicit feedback, text, tags | Knowledge-graph-oriented data |
| **Music4All-Onion** | Music audio/video/metadata | 109,269 music pieces |

# Catalogue: books, games, images, and jokes
| Dataset | Primary signal | Detail |
| :--- | :--- | :--- |
| **Book-Crossing** | Explicit + implicit book ratings | Demographics included |
| **GoodReads** | Shelves, ratings, reviews | Multiple interaction levels |
| **Steam** | Reviews, play hours, game metadata | 7.79M reviews |
| **Anime** | User ratings | MyAnimeList preferences |
| **Pinterest** | Image/social recommendation | Content-based image recommendation |
| **Jester** | Anonymous joke ratings | Jester recommender data |

# Catalogue: learning and health data
| Dataset | Primary signal | Detail |
| :--- | :--- | :--- |
| **KDD Cup 2010** | Student exercise attempts | Educational data mining |
| **EndoMondo** | Workout and sensor logs | Health-sensitive sequential data |

# Catalogue: web, news, reviews, and apparel
| Dataset | Primary signal | Detail |
| :--- | :--- | :--- |
| **Phishing Websites** | Website features + labels | Classification, not user-item recommendation |
| **Behance** | Likes + image data | Anonymised art-community subset |
| ★ **MIND-small** | News impressions + article text | Detail slide uses a manageable subset |
| **BeerAdvocate / RateBeer** | Multi-aspect beer reviews | Taste, look, feel, smell |
| **ModCloth / RentTheRunway** | Clothing-fit feedback | Fit measurements |
| **Adult** | Census attributes + income label | Classification; not a recommender dataset |

# Catalogue: course-text datasets beyond RecBole
| Dataset | Primary signal | Detail |
| :--- | :--- | :--- |
| ★ **Adressa** | News reading sessions + articles | 16-day processed benchmark |
| **Reddit** | Community / session interactions | 1.14M sessions in course book table |
| ★ **CiteULike-a** | User-paper saves + tags | Scholarly recommendation |
| **BBC News (LLM book)** | News articles + text | Text-based recommendation example |
| **Amazon Fashion (LLM book)** | Reviews, images, product text | A scoped Amazon 2023 category |

# 03 Selected Dataset Details ⭐

# MovieLens 1M
* **INTERACTIONS:** 1,000,209 (6,040 users · 3,706 movies)
* **FOOTPRINT:** 6 MB ZIP (Under 1 GB RAM for standard experiments)
* **DATA:** Ratings (1-5 stars, timestamps, genres, demographics)

**Best fit:** transparent CF or ranking baselines; factorisation, neural CF, and genre-aware hybrids.

**Papers:** Harper & Konstan (2015), *The MovieLens Datasets*; Koren, Bell & Volinsky (2009), *Matrix Factorization Techniques for Recommender Systems*.
**URL:** https://grouplens.org/datasets/movielens/1m/

> **Warning**
> The demographic fields are coarse and historic. Do not claim they represent contemporary audiences or use them casually as protected-attribute ground truth.

# Amazon Reviews 2023 - select one category
* **CATALOGUE SCALE:** 571.5M (Reviews across 44 categories)
* **FOOTPRINT:** Varies (Use one category + a documented k-core)
* **DATA:** Hybrid (Ratings, reviews, metadata, images by category)

**Best fit:** item cold start, review-aware recommendation, retrieval from product text, or an LLM-assisted explanation study.

**Papers:** McAuley et al. (2015), *Image-based Recommendations*; Hou et al. (2024), *Amazon Reviews 2023*.
**URL:** https://amazon-reviews-2023.github.io/

> **Warning**
> Choose a category before downloading. Reviews are not neutral preference labels; helpfulness, popularity, and seller incentives can bias what is observed.

# Yelp Open Dataset
* **REVIEWS:** 6.99M (150,346 businesses · 11 metro areas)
* **FOOTPRINT:** ~4.4 GB ZIP (Plan 8-16 GB RAM for full JSON processing)
* **DATA:** Reviews + POIs (Text, stars, attributes, hours, check-ins)

**Best fit:** business/POI recommendation using text and metadata; city-level temporal split; exposure and representation analysis.

**Source / papers:** Yelp (2022), *Yelp Open Dataset*; He et al. (2017), *Neural Collaborative Filtering* (a useful rating-ranking baseline).
**URL:** https://business.yelp.com/data/resources/open-dataset/

# MIND-small
* **BEHAVIOUR LOGS:** 230,117 (94,057 users · 65,238 news articles)
* **CLICKS:** 347,727 (Use provided train / dev divisions)
* **DATA:** Impressions + text (Title, abstract, category, entities, click history)

**Best fit:** candidate ranking, news encoders, sequential interests, and ranking metrics such as AUC, MRR, and nDCG.

**Papers:** Wu et al. (2020), *MIND*; Wu et al. (2019), *NAML: Neural News Recommendation with Attentive Multi-View Learning*.
**URL:** https://msnews.github.io/

> **Warning**
> Non-clicked impressions are observed non-clicks, not certain dislikes. Discuss position/exposure bias, short news life cycles, and misinformation harms.

# Diginetica
* **SESSIONS:** 204,789 (993,483 click events)
* **ITEMS:** 43,136 (Average session length: 4.85)
* **DATA:** Search sessions (Clicks, queries, product metadata, prices)

**Best fit:** next-item ranking on short anonymous sessions; compare item-kNN, GRU4Rec, and Transformer/attention models.

**Papers:** Kotsifakos et al. (2019), *Diginetica: A Dataset for Session-based Recommendation*; Hidasi et al. (2016), *Session-based Recommendations with Recurrent Neural Networks*.
**URL:** https://competitions.codalab.org/competitions/11161

# YOOCHOOSE / RecSys Challenge 2015
* **SESSIONS:** 7.98M (31.71M click/buy events)
* **ITEMS:** 37,486 (182 days · mean length 3.97)
* **DATA:** Sessions (E-commerce clickstream and purchase signal)

**Best fit:** an ambitious sequential project. Work on a chronologically defined subset before scaling.

**Papers:** Hidasi et al. (2016), *Session-based Recommendations with RNNs*; Tan et al. (2016), *Improved Recurrent Neural Networks for Session-based Recommendations*.
**URL:** https://2015.recsyschallenge.com/challenge.html

> **Warning**
> This is a session dataset, not a long-term personalised-profile dataset. Avoid claims about stable user preference or demographic fairness that the data cannot support.

# Retailrocket
* **EVENTS:** 2.76M (1.41M visitors · 235,061 items)
* **FOOTPRINT:** ~340 MB CSV (Plan 2-4 GB RAM after parsing)
* **DATA:** Events (View, add-to-cart, transaction, timestamps)

**Best fit:** event-type-aware sequential recommendation; compare whether carts/purchases should be weighted differently from views.

**Papers:** Hidasi & Karatzoglou (2018), *Recurrent Neural Networks with Top-k Gains*; Quadrana, Cremonesi & Jannach (2017), *Sequence-Aware Recommender Systems*.
**URL:** https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

# Adressa
* **SESSIONS:** 982,210 (2.65M click events in a 16-day processed version)
* **ITEMS:** 13,820 (Short sessions: mean length 2.7)
* **DATA:** News sessions (Reading events, article metadata/content by release)

**Best fit:** short-horizon news recommendation and temporal evaluation; compare session-only and text-aware ranking.

**Papers:** Gulla et al. (2017), *The Adressa Dataset for News Recommendation*; Karimi et al. (2018), *News Recommender Systems*.
**URL:** https://reclab.idi.ntnu.no/dataset/

> **Warning**
> News preference is volatile and politically consequential. Preserve time order, avoid leaking future article metadata, and analyse publisher/topic concentration.

# Last.fm 2K
* **INTERACTIONS:** 92,834 (1,892 users · 17,632 artists)
* **FOOTPRINT:** Small ZIP (Laptop-friendly; under 1 GB RAM)
* **DATA:** Implicit + social (Listening counts, tags, user friendships)

**Best fit:** tag-aware, social, and implicit-feedback recommendation; investigate diversity beyond head artists.

**Papers:** Cantador, Brusilovsky & Kuflik (2011), *2nd HetRec Workshop*; Kula (2015), *Metadata Embeddings for User and Item Cold-start Recommendations*.
**URL:** https://grouplens.org/datasets/hetrec-2011/

# Gowalla
* **CHECK-INS:** 6.44M (196,591 users · Feb 2009-Oct 2010)
* **SOCIAL GRAPH:** 950,327 (Friendship edges; 107,092 users appear in check-ins)
* **DATA:** Spatiotemporal (Time, latitude, longitude, POI, friendship)

**Best fit:** next-POI recommendation, geographical influence, or social/context-aware ranking.

**Papers:** Cho, Myers & Leskovec (2011), *Friendship and Mobility*; Ye, Yin & Lee (2011), *Exploiting Geographical Influence for Collaborative POI Recommendation*.
**URL:** https://snap.stanford.edu/data/loc-gowalla.html

> **Warning**
> Precise location traces are sensitive. Minimise retained coordinates, avoid re-identification claims, and do not visualise individual mobility paths without a strong reason.

# Epinions
* **RATINGS:** 664,824 (49,290 users · 139,738 items)
* **TRUST LINKS:** 487,183 (Directed user-user edges)
* **DATA:** Ratings + trust (Historic consumer-review community data)

**Best fit:** trust propagation, social regularisation, robustness, and cold-start comparison against a rating-only baseline.

**Papers:** Massa & Avesani (2006), *Trust-aware Recommender Systems*; Guo, Zhang & Yorke-Smith (2015), *TrustSVD*.
**URL:** https://www.cse.msu.edu/~tangjili/trust.html

> **Warning**
> Trust links are neither ground-truth expertise nor similarity. The data are historical, so frame claims as benchmark evidence rather than contemporary behaviour.

# CiteULike-a
* **USER-PAPER SAVES:** 204,986 (5,551 users · 16,980 papers)
* **FOOTPRINT:** Small (Laptop-friendly after text/tag filtering)
* **DATA:** Implicit + tags (Paper saves, tags, and textual content by release)

**Best fit:** scholarly-paper recommendation, collaborative topic models, tag-aware retrieval, or an LLM-based text encoder against a clean baseline.

**Papers:** Wang & Blei (2011), *Collaborative Topic Modeling for Recommending Scientific Articles*; Nallapati et al. (2008), *Joint Latent Topic Models for Text and Citations*.
**URL:** https://github.com/js05212/citeulike-a

# 04 Common Practices

# Choose a dataset by the claim you can defend
1. **Define the task:** Who receives what recommendation, from which signal, at what time?
2. **Lock the split:** Use a chronological split for sequential, news, or context-sensitive claims.
3. **Establish baselines:** Start with popularity and a classical model before a neural/LLM system.
4. **Evaluate the interface:** Use ranking metrics for top-k recommendation; inspect coverage and diversity too.
5. **Analyse failure:** Segment by activity level, item popularity, time, and relevant content groups.
6. **Address people:** Document licence, provenance, privacy, exposure, representation, and likely harms.

# A reproducibility checklist
* Record the exact source URL, download date, licence/terms, version, and checksum where supplied.
* Define every filter: event types retained, k-core threshold, sessionisation rule, and minimum sequence length.
* Split before fitting any data-dependent representation; do not let future interactions or item text leak into training.
* Publish a configuration for popularity, ItemKNN/BPR/MF, and the advanced model on the same split.
* Report resource use: sampled records, download size, GPU/CPU, RAM, training time, and random seeds.
* Separate an observed non-click from a negative preference unless the task design justifies the assumption.
