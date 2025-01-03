# QUACER-C: QUANTITATIVE CERTIFICATION OF KNOWLEDGE COMPREHENSION IN LLMS  

Isha Chaudhary1, Vedaant V. Jain1 & Gagandeep Singh1,2 1University of Illinois Urbana-Champaign, USA 2VMware Research, USA {isha4, vvjain3, ggnds}@illinois.edu  

# ABSTRACT  

Large Language Models (LLMs) have demonstrated impressive performance on several benchmarks. However, traditional studies do not provide formal guarantees on the performance of LLMs. In this work, we propose a novel certification framework for LLM, QuaCer-, wherein we formally certify the knowledgecomprehension capabilities of popular LLMs. Our certificates are quantitative — they consist of high-confidence, tight bounds on the probability that the target LLM gives the correct answer on any relevant knowledge comprehension prompt. Our certificates for the Llama, Vicuna, and Mistral LLMs indicate that the knowledge comprehension capability improves with an increase in the number of parameters and that the Mistral model is less performant than the rest in this evaluation.  

# 1 INTRODUCTION  

Large Language Models (LLMs) have demonstrated human-level performance for several real-world downstream tasks (Yang et al., 2023; Liang et al., 2023; Bommasani et al., 2022). An important use case for LLMs is knowledge comprehension (Lazaridou et al., 2022; Khattab et al., 2023), i.e., they are often used to summarize long texts, respond to user queries based on the context, and serve as adaptive task decomposers for reasoning based retrieval augmented generation tasks (Yao et al., 2023). Large context windows of millions of tokens in models like Gemini v1.5 (Gemini Team, 2024) reduce reliance on large knowledge corpora of RAG systems and parametric knowledge held by LLMs. Users increasingly provide extensive references during inference to guide responses. This makes analyzing the knowledge comprehension and reasoning capabilities of popular LLMs crucial.  

There have been several studies for benchmarking the performance of LLMs for knowledge comprehension using multi-hop question-answering datasets (Liang et al., 2023; Chen et al., 2021; Yang et al., 2018; Wang et al., 2023; Trivedi et al., 2022; Tang & Yang, 2024). These datasets consist of questions that require several reasoning steps to get the correct answer. However, prior studies lack a formal analysis of the knowledge comprehension capability of the LLM. Our work aims to bridge this gap by introducing a novel formal certification method, QuaCer-C, for quantitatively certifying the knowledge comprehension capability of LLMs.  

Key challenges. We face the following challenges when developing our formal certification framework for LLMs. (1) We need to find a formal expression capturing the knowledge comprehension property that is amenable to certification. Such an expression should capture the diverse inputs of the property that the LLM should correctly handle. (2) Giving provable guarantees on LLMs is a hard, open problem, due to the high number of parameters of the models, for which the traditional certifiers (Singh et al., 2019; Shi et al., 2020; Bonaert et al., 2021) would lose significant precision leading to inconclusive analysis. Moreover, failure examples where the desirable property does not hold are fairly easy to construct for popular LLMs by appropriate prompt tuning (Xu et al., 2024), making binary certificates that indicate whether an LLM satisfies a specification trivial and pointless.  

Our approach. We first conceptualize the knowledge comprehension property as a formal specification using a knowledge graph. We develop quantitative properties such that we can obtain a measure for the knowledge-comprehension capability of the target LLM. We then propose a modelagnostic, black-box quantitative certification approach, QuaCer-C, which circumvents the issues that traditional approaches have with the number of parameters in LLMs and can even work for closedsource LLMs. QuaCer-C estimates high-confidence bounds on the quantitative property based on a sufficient number of queries, using the confidence intervals (CLOPPER & PEARSON, 1934).  

# Contributions. We make the following contributions:  

1. We specify the knowledge comprehension property desirable from the LLM responses as a formal expression. Our specifications use popular knowledge graphs such as Wikidata5m (Wang et al., 2021) that are augmented with supporting information about each of their entities. The specifications represent a large set of knowledge comprehension prompts with their respective correct answers that are expected from any target LLM.   
2. We model certification in a target LLM as a probability estimation problem and leverage Clopper-Pearson confidence intervals to generate provable, high-confidence bounds on the quantitative property of interest. Our implementation and data are open-sourced at https://github.com/uiuc-focal-lab/QuaCer-C.   
3. We generate the proposed certificates for the popular LLMs: Llama 7B and 13B, Vicuna 7B and 13B, and Mistral-7B. We observe that as the number of model parameters increases, the knowledge comprehension capability of the LLM improves. On comparing the different model classes, we see Mistral performing worse than Llama and Vicuna.  

# 2 CERTIFYING KNOWLEDGE COMPREHENSION  

Consider an LLM $\mathcal{L}:\chi\to\ y$ that takes prompts $p\in\chi$ as input and generates responses $r\,\in\mathcal{V}$ , where $\chi$ and $\boldsymbol{\wp}$ are sets of all possible prompts to the LLM and the responses generated by it, respectively. We formally assess $\mathcal{L}$ for its capability to comprehend knowledge given in its prompt and answer a multi-hop query involving multiple reasoning steps based on the given knowledge in the prompt. Figure 1 illustrates the QuaCer-C certification framework. We define the knowledge comprehension property of LLMs next.  

# 2.1 FORMAL SPECIFICATION  

The property is defined over a knowledge graph $\mathcal{G}\,=\,(\mathcal{V},\mathcal{E})$ having nodes $\nu$ that correspond to distinct entities (e.g., name of a country) that are augmented with textual information about them and directed edges $\mathcal{E}$ between pairs of nodes that are labeled by the relation between the nodes. There can be multiple names aka aliases to identify each entity and each label for edges. An example of such a knowledge graph is Wikidata5m (Wang et al., 2021) which consists of nodes for entities having Wikipedia pages, along with the abstracts from the respective pages. Two nodes $(v_{1},v_{2})$ are connected by a labeled edge if there is a link in the supporting document for $v_{1}$ to that for $v_{2}$ .  

Definition 2.1. (Path in a Knowledge Graph). A path $\Pi=[v_{1},v_{2}\ldots,v_{l}]$ is an ordered collection of nodes in a given knowledge graph $\mathcal{G}$ , where $l>1$ , such that $\forall i\in[l-1]$ . $v_{i}\in\mathcal{V}$ and $(v_{i},v_{i+1})\in\mathcal{E}$ .  

Next, we describe two primitive functions — prompt constructor and response checker — with which we specify the knowledge comprehension property. Let $\Pi_{\mathcal{G}}$ denote all paths in $\mathcal{G}$ , a prompt constructor $\mathcal{Q}_{\mathcal{G}}$ takes a path (Definition 2.1) $\Pi\,\in\,\Pi_{\mathcal{G}}$ of finite but arbitrary length as input, and randomly generates a query-based prompt and the correct answer. Each prompt produced from a given path is a concatenation of some textual information about nodes and a query. The textual information consists of randomly organized supporting context for all the nodes in $\Pi$ and other randomly selected distractor texts from nodes not in the path. The query in the prompt is on the entity corresponding to the tail of $\Pi$ and it describes the rest of the path. The correct answer to the prompt is an alias of the tail of $\Pi$ . The specific prompt constructor function is user-defined and can determine the difficulty level of the prompts with which the target LLM $\mathcal{L}$ is certified. A stochastic prompt constructor enables us to develop a general specification that takes the effect of common aliases on $\mathcal{L}$ into account. A response checker $\mathcal{C}_{\mathcal{L}}(p,r)$ is a boolean function that evaluates to true if $\mathcal{L}$ ’s response for prompt $p$ matches any of the aliases of the given node $r$ .  

Our knowledge comprehension property requires that the target LLM $\mathcal{L}$ correctly answers queries contained in the prompts generated by the prompt constructor $\mathcal{Q}_{\mathcal{G}}$ on all paths in $\Pi_{\mathcal{G}}$ , as identified  

![](images/0058f3663e834a540c48520d40cbc321e3fa490d2bc01b2e6b5915080a70f718.jpg)  

Figure 1: Overview of QuaCer-C. (a) A knowledge graph $\mathcal{G}$ pivoted on some node, in this case the ’Paul Sophus Epstein’. (b) A randomly chosen path originating at the pivot node from the various possibilities in $\mathcal{G}$ . (c) A prompt created by our prompt constructor using the selected path and context from the Wikidata5m corpus of the entities (nodes) involved, along with a distractor context and the final query. (d) The target LLM’s output to the prompt, validated using the response checker. (e) Certifier obtains bounds on the probability of correct response and we iterate till the range of the bounds falls below a threshold. (f) The final bounds contained in the certificate.  

by the response checker $\mathcal{C}$ . We reformulate the property to make it quantitative by measuring the probability that the target LLM $\mathcal{L}$ correctly answers the query in a prompt over a randomly selected path from $\Pi_{\mathcal{G}}$ (Eq. 1), where $D i s t(\Pi_{\mathcal{G}})$ is a user-defined probability distribution over all the paths $\Pi\in\Pi_{\mathcal{G}}$ . A higher probability value indicates better knowledge comprehension capability of $\mathcal{L}$ .  

$$
P r_{\Pi\sim D i s t(\Pi_{\mathcal{G}})}\mathcal{C}_{\mathcal{L}}(\mathcal{Q}_{\mathcal{G}}(\Pi)))
$$  

From a practical standpoint, longer paths can become semantically meaningless, and thus shorter path lengths are considered in popular multi-hop question-answering datasets such as (Yang et al., 2018; Chen et al., 2021; Trivedi et al., 2022). Thus, we upper-bound the lengths of paths, which we define as the number of nodes in the path, used for certification by a hyperparameter $\rho$ . Let $\Pi_{\mathcal{G},\rho}\subseteq$ $\Pi_{\mathcal{G}}$ denote all the paths, where each path has a maximum length $\rho$ . Therefore, our knowledge comprehension property becomes Eq. 2 for any given maximum path length.  

$$
P r_{\Pi\sim D i s t(\Pi_{\mathcal{G},\rho})}\mathcal{C}_{\mathcal{L}}(\mathcal{Q}_{\mathcal{G}}(\Pi))
$$  

Global certification over the entire $\mathcal{G}$ involves a large number of queries and may not scale for realworld graphs having millions of nodes. Hence, we scope the probabilistic property for $\mathcal{L}$ (Eq. 2) into a local subgraph level property, wherein each subgraph $\mathcal G(v)$ is defined by a randomly selected pivot node $v$ and consists of all paths $\Pi_{v}$ originating from $v$ having maximum length of $\rho$ . Hence, the local probabilistic property becomes Eq. 3.  

$$
P r_{\Pi\sim D i s t(\Pi_{v})}\mathcal{C}_{\mathcal{L}}(\mathcal{Q}_{\mathcal{G}}(\Pi))
$$  

2.2 CERTIFICATION METHOD  

QuaCer-C certifies the target LLM $\mathcal{L}$ by computing an interval $[p_{l},p_{u}]$ containing the value of the property (Eq. 3) for a given pivot node $v$ , prompt constructor $\mathcal{Q}_{\mathcal{G}}$ , response checker $\mathcal{C}$ , and distribution $D i s t(\Pi_{v})$ of paths in the subgraph of interest. We want tight intervals, i.e., a low value of $p_{u}-p_{l}$ , for precise certification. For reliability, we construct intervals that bound the value of interest with high confidence. We determine bounds with higher confidence than the user-specified confidence level $1-\delta$ , where $\delta$ is a small positive constant, using the Clopper-Pearson confidence intervals (CLOPPER & PEARSON, 1934; Brown et al., 2001; Kurz et al., 2014; Collins, 2010) based on samples of $\mathcal{L}$ ’s responses. The bounds obtained are conservative, i.e., they are lower and upper bounds for the target property’s value $p$ with a confidence of at least $1-\delta$ .  

$$
P r[p_{l}\leq p\leq p_{u}]\geq1-\delta
$$  

We model the value $p$ of the property in 3 as the probability of success of the underlying boolean random variable $\mathcal{R}=\mathcal{C}_{\mathcal{L}}(\mathcal{Q}_{\mathcal{G}}(\Pi))$ , a function of the random variable $\Pi\sim\mathit{D i s t}(\Pi_{v})$ . We model $\mathcal{R}$ as a Bernoulli random variable with an unknown, constant probability of success, denoted by $p$ . This assumption is based on the expectation that specifications, as defined on local subgraphs, will capture similar queries that elicit similar responses from $\mathcal{L}$ and, consequently, a consistent probability of success across samples. To compute high-confidence tight bounds on $p$ , we make $n$ random observations of $\mathcal{R}$ , out of which we obtain $k\leq n$ successes, where success is equivalent to getting a true value for $\mathcal{R}$ . With the $n$ observations and $1\!-\!\delta$ confidence, we use the Clopper-Pearson confidence intervals $p$ with upper and lower bound $p_{u},p_{l}$ respectively. We make a dynamic number of observations $n$ till the range between the bounds $p_{u}-p_{l}$ falls below a prespecified threshold $\alpha$ , which should be small for precise certificates. However, $n$ increases on decreasing $\alpha$ , therefore the value of $\alpha$ can be tuned according to the certification time budget and desired precision.  

# 3 EXPERIMENTS  

We generate the proposed certificate for popular LLMs and derive insights into their knowledge comprehension capability when compared with each other.  

Experimental setup. We certify the 7B and 13B parameter Llama-2 models (Touvron et al., 2023) that have been instruction-tuned and safety-aligned, Vicuna 7B, 13B (Chiang et al., 2023), and Mistral 7B-Instruct-v0.2 (Jiang et al., 2023). We use Wikidata5m (Wang et al., 2021) as our base knowledge graph after preprocessing (check Appendix B for details) from which we develop a test set of 50 specifications, each pivoted at a randomly selected node in Wikidata5m having a high number of possible paths, making enumerative certification (where all possibilities are tested for satisfaction of the specification) impractical. We select the largest maximum path length parameter $\rho~=~5$ , which could still form a sensible query. A path is sampled from our distribution over all paths in a subgraph by first uniformly randomly selecting a path length from $[1,\rho]$ and then uniformly randomly selecting a path having the chosen path length. We detail our prompt constructor which also samples from the distribution of all paths in the graph in Appendix A.1. We evaluate the responses of LLMs using another LLM (Mistral 7B) as our response checker function. The detailed design and evaluation of our response checker in identifying correct and wrong responses is presented in Appendix A.2. QuaCer-C generates certificates with confidence $1-\delta\bar{=}\,95\%$ and certify till the range of the bounds falls below a threshold $\alpha=0.1$ . We also keep a maximum cap on the number of samples, which we fix as 500 samples. We report the bounds even with a range higher than $\alpha$ , after 500 samples. We conduct our experiments on 4 A100 GPUs hosting the LLMs.  

Certificates. QuaCer-C generates certificates providing tight, high-confidence lower and upper bounds on the probability of a correct LLM response to a random prompt obtained from the prompt constructor applied on a given subgraph. We report the average value of each bound, over the test set of properties that QuaCer-C certifies for each LLM, in Table 1. We also report the average empirical probability (the ratio of correct responses to the total number of prompts, $n$ , for each certificate), averaged over the test set.  

We also study the robustness of the target LLMs in correctly answering random queries generated by perturbing the aliases in a query corresponding to a path. The average robustness over all the paths considered in the certificates is $(60\pm5)\%$ for each target model.  

Discussion. The certificates provide provable bounds on the knowledge comprehension capability of a target LLM in an average case. QuaCer-C’s certification results indicate an improvement in the knowledge comprehension capability as the number of parameters increases. This is evident for both the Llama and Vicuna models with their average empirical values and the average values of the bounds. Moreover, we observe that Mistral performs worse than the Llama and Vicuna models.  

Table 1: Certification results for different LLMs   

![](images/153f53d7d076293ab3dd8781194ea9af6031468c151c033dfc9b1d837228fbf0.jpg)  

# REFERENCES  

Rishi Bommasani, Drew A. Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, Erik Brynjolfsson, Shyamal Buch, Dallas Card, Rodrigo Castellon, Niladri Chatterji, Annie Chen, Kathleen Creel, Jared Quincy Davis, Dora Demszky, Chris Donahue, Moussa Doumbouya, Esin Durmus, Stefano Ermon, John Etchemendy, Kawin Ethayarajh, Li Fei-Fei, Chelsea Finn, Trevor Gale, Lauren Gillespie, Karan Goel, Noah Goodman, Shelby Grossman, Neel Guha, Tatsunori Hashimoto, Peter Henderson, John Hewitt, Daniel E. Ho, Jenny Hong, Kyle Hsu, Jing Huang, Thomas Icard, Saahil Jain, Dan Jurafsky, Pratyusha Kalluri, Siddharth Karamcheti, Geoff Keeling, Fereshte Khani, Omar Khattab, Pang Wei Koh, Mark Krass, Ranjay Krishna, Rohith Kuditipudi, Ananya Kumar, Faisal Ladhak, Mina Lee, Tony Lee, Jure Leskovec, Isabelle Levent, Xiang Lisa Li, Xuechen Li, Tengyu Ma, Ali Malik, Christopher D. Manning, Suvir Mirchandani, Eric Mitchell, Zanele Munyikwa, Suraj Nair, Avanika Narayan, Deepak Narayanan, Ben Newman, Allen Nie, Juan Carlos Niebles, Hamed Nilforoshan, Julian Nyarko, Giray Ogut, Laurel Orr, Isabel Papadimitriou, Joon Sung Park, Chris Piech, Eva Portelance, Christopher Potts, Aditi Raghunathan, Rob Reich, Hongyu Ren, Frieda Rong, Yusuf Roohani, Camilo Ruiz, Jack Ryan, Christopher Re´, Dorsa Sadigh, Shiori Sagawa, Keshav Santhanam, Andy Shih, Krishnan Srinivasan, Alex Tamkin, Rohan Taori, Armin W. Thomas, Florian Trame\`r, Rose E. Wang, William Wang, Bohan Wu, Jiajun Wu, Yuhuai Wu, Sang Michael Xie, Michihiro Yasunaga, Jiaxuan You, Matei Zaharia, Michael Zhang, Tianyi Zhang, Xikun Zhang, Yuhui Zhang, Lucia Zheng, Kaitlyn Zhou, and Percy Liang. On the opportunities and risks of foundation models, 2022.  

Gregory Bonaert, Dimitar I. Dimitrov, Maximilian Baader, and Martin Vechev. Fast and precise certification of transformers. In Proceedings of the 42nd ACM SIGPLAN International Conference on Programming Language Design and Implementation, PLDI 2021, pp. 466–481, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450383912. doi: 10.1145/3453483.3454056. URL https://doi.org/10.1145/3453483.3454056.  

Lawrence D. Brown, T. Tony Cai, and Anirban DasGupta. Interval Estimation for a Binomial Proportion. Statistical Science, 16(2):101 – 133, 2001. doi: 10.1214/ss/1009213286. URL https://doi.org/10.1214/ss/1009213286.  

Wenhu Chen, Hanwen Zha, Zhiyu Chen, Wenhan Xiong, Hong Wang, and William Wang. Hybridqa: A dataset of multi-hop question answering over tabular and textual data, 2021.  

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with $90\%*$ chatgpt quality, March 2023. URL https: //lmsys.org/blog/2023-03-30-vicuna/.  

C. J. CLOPPER and E. S. PEARSON. THE USE OF CONFIDENCE OR FIDUCIAL LIMITS ILLUSTRATED IN THE CASE OF THE BINOMIAL. Biometrika, 26(4):404–413, 12 1934. ISSN 0006-3444. doi: 10.1093/biomet/26.4.404. URL https://doi.org/10.1093/biomet/ 26.4.404.  

Joseph Collins. Binomial distribution: Hypothesis testing, confidence intervals (ci), and reliability with implementation in s-plus. pp. 43, 06 2010.  

Google Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024.  

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2021.  

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Le´lio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothe´e Lacroix, and William El Sayed. Mistral 7b, 2023.  

Omar Khattab, Keshav Santhanam, Xiang Lisa Li, David Hall, Percy Liang, Christopher Potts, and Matei Zaharia. Demonstrate-search-predict: Composing retrieval and language models for knowledge-intensive nlp, 2023.  

Daniel Kurz, Horst Lewitschnig, and Ju¨rgen Pilz. Decision-theoretical model for failures which are tackled by countermeasures. IEEE Transactions on Reliability, 63(2):583–592, 2014. doi: 10.1109/TR.2014.2315952.  

Angeliki Lazaridou, Elena Gribovskaya, Wojciech Stokowiec, and Nikolai Grigorev. Internetaugmented language models through few-shot prompting for open-domain question answering, 2022.  

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Re´, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, and Yuta Koreeda. Holistic evaluation of language models, 2023.  

Zhouxing Shi, Huan Zhang, Kai-Wei Chang, Minlie Huang, and Cho-Jui Hsieh. Robustness verification for transformers, 2020.  

Gagandeep Singh, Timon Gehr, Markus Pu¨schel, and Martin Vechev. An abstract domain for certifying neural networks. Proc. ACM Program. Lang., 3(POPL), jan 2019. doi: 10.1145/3290354. URL https://doi.org/10.1145/3290354.  

Yixuan Tang and Yi Yang. Multihop-rag: Benchmarking retrieval-augmented generation for multihop queries, 2024.  

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023.  

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. MuSiQue: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 2022.  

Jinyuan Wang, Junlong Li, and Hai Zhao. Self-prompted chain-of-thought on large language models for open-domain multi-hop reasoning, 2023.  

Xiaozhi Wang, Tianyu Gao, Zhaocheng Zhu, Zhengyan Zhang, Zhiyuan Liu, Juanzi Li, and Jian Tang. Kepler: A unified model for knowledge embedding and pre-trained language representation. Transactions of the Association for Computational Linguistics, 9:176–194, 2021.   
Xilie Xu, Keyi Kong, Ning Liu, Lizhen Cui, Di Wang, Jingfeng Zhang, and Mohan Kankanhalli. An LLM can fool itself: A prompt-based adversarial attack. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id $=$ VVgGbB9TNV.   
Xianjun Yang, Yan Li, Xinlu Zhang, Haifeng Chen, and Wei Cheng. Exploring the limits of chatgpt for query or aspect-based text summarization, 2023.   
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018.   
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models, 2023.  

# A CERTIFICATION FRAMEWORK COMPONENTS  

A.1 PROMPT CONSTRUCTOR  

We use a prompt constructor designed to generate queries that facilitate a large language model’s (LLM) navigation through a knowledge graph. The navigation process adheres to the constraint, $\rho$ which we will refer to as $k$ in this section, that limits the maximum number of inferential steps the LLM can take in its reasoning path. Our approach emphasizes explicit multi-hop reasoning while operating within the specified step count.  

Algorithm 1 outlines the construction process. The input for the algorithm in the query path we get from employing the PathGeneration detailed in Algorithm 2. This path is then transformed into a guiding query by the QueryGenerationPath function 3 that guides the LLM’s reasoning. To provide essential context, the FormContext function gathers relevant Wikidata5m entries for all entities in the path and includes a distractor context to enhance task complexity. Additionally, during prompt generation, aliases for relations and entity in question are chosen at random using the GetAlias function. In order to support alias substitution and provide relevant information, the alias information for the target entity is added to the query context. The GetAlias function returns a randomly chosen alias(alternative name) for an input using the data from the wikidata5m dataset which is the basis of the knowledge graph we are using.  

This procedure of prompt construction emphasizes traversing a graph within a constrained step count to construct a query.  

![](images/74cdbf0a0e4ef8d82986845ee46feea36214f72344a019de62362e26c148e898.jpg)  

The path generation algorithm is outlined in 2. The algorithm commences by using the source node on which the subgraph is pivoted and subsequently chooses a random value (k choice) between 1 and $\boldsymbol{\mathrm{k}}$ , inclusive. This value dictates the length of the reasoning path. Next, a depth-first search generates a path of length k choice from the source node.  

QUERYGENERATIONPATH  

![](images/aab0e714b9b1e7d60cb9f7d5cbf6ed6dba8176aefdb5f9c4186fe2e1b9af6bb2.jpg)  

The function initializes the query if cur query is empty. If the path contains only a single node, that node is appended to the query. Otherwise, the function appends the descriptive relation between the last two vertices of the path to the current query and continues recursively with the remaining path.  

EXAMPLE  

Consider the following scenario within a graph: a path from Chandler Bing to Matthew Perry, then to 19 August 1969, with the relationships between Chandler Bing and Matthew Perry defined as ”Actor” and between Matthew Perry and 19 August 1969 as ”Birth Date.” The algorithm would construct the query: ”What is the birth date of the actor of Chandler Bing?” This query guides the LLM to reason through the defined relationships, demonstrating the algorithm’s practical application.  

A.2 RESPONSE CHECKER FUNCTION  

To ensure the correctness of answers generated by a language model (LLM A), we employ a twostep validation process.  

# Step 1: Alias Validation  

Firstly, we leverage the Wikidata5m dataset to compile a comprehensive list of aliases associated with the answer entity. If any of these aliases are found within LLM A’s response, the answer is directly approved else, we move to step 2.  

# Step 2: Evaluation with LLM Checker  

Secondly, if no aliases match, we engage a separate language model (LLM Checker) to validate LLM A’s output.  

A.2.1 EVALUATION WITH LLM CHECKER  

LLM Checker’s verification process employs a dedicated checker function that takes in three inputs:  

• The original query: Helps LLM Checker understand the question.   
• The correct answer: Serves as the ground truth.   
• LLM A’s generated answer: The response to be assessed.  

We format the evaluation task for LLM Checker as a question-answering problem. This allows LLM Checker to leverage its language comprehension and parametric knowledge to determine the alignment between LLM A’s answer and the ground truth.  

We provide LLM Checker with the following context for evaluation.  

• ”You are a helpful assistant. Your task is to assess inputs comprising a question, its correct answer, and an answer provided by a model. You should affirm with ’yes’ if the model’s answer is semantically equivalent to the correct answer, or ’no’ if it is not.”  

Additionally, to ensure deterministic results we set the temperature of LLM Checker to be 0.  

Prompt Structure The assessment prompt provided to LLM Checker follows this template: ”Question: {query} Ground Truth: {correct answer}. Model Answer: {LLM A answer}.”  

For practical implementation, we use a Mistral-7B-Instruct-v0.2 as the checker LLM.  

This method is useful because this allows better evaluation of correct answers in various forms such as synonyms. An example where this is useful is for the following: ’question’: ’Which of the following best describes the structure that collects urine in the body?’, ’correct answer’: ’Bladder’, model answer: ’Urinary Bladder’.  

LLM Checker Function Evaluation We validate the checker function’s efficacy using 1000 samples from the MMLU (Hendrycks et al., 2021) dataset, divided equally between perturbed correct answers and incorrect answers. Perturbed answers are the same as the correct answers but not verbatim. To generate perturbed answers, we separately use the Gemini Pro model API to generate perturbed answers for question-answer pairs when possible. To do so, we give the model, the question, the correct answer, and the other options from the MMLU dataset. The model may or may not decide to perturb the answer. Additionally, to generate a wrong answer dataset, we simply use other options from the MMLU dataset that are not the correct answer for the given question. The accuracy of the Mistral 7b Instruct v2 with default settings is $85\%$ when we set the temperature to 0.  

B PREPROCESSING THE WIKIDATA5M KNOWLEDGE GRAPH  

To ensure the generation of unambiguous queries and support the certification process, we preprocess the wikidata5m dataset.  

1. Relation Filtering: We remove relations such as ’instance of’, ’subclass of’, and ’part of’ due to their inherent potential for ambiguity in query formulation.   
2. Unique Relation Enforcement: We retain only relations that establish a unique connection from a given entity. This means any relations where an entity might connect to multiple other entities (e.g., a ’parent of’ relation) are disregarded. This filtering step is crucial for longer multi-hop queries, preventing scenarios where different paths might have conflicting properties.   
3. Reciprocal Support Verification: We establish a requirement for reciprocal support within the descriptive text associated with entities. If entity A has a relation to entity B, either entity B’s supporting text must mention entity A or vice versa. Relations failing this check are discarded, ensuring that the context provided by the prompt generator will likely contain sufficient information to resolve the query.   
4. Unicode Conversion: For consistency within our experiments, we convert all text containing Unicode characters into their respective ASCII approximations.  

Rationale: These preprocessing steps promote the generation of clear, well-defined queries while supporting the integrity of the certification framework.  