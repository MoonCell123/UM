# Fusion of Multi-scale Heterogeneous Pathology Foundation Models for Whole Slide Image Analysis

Zhidong Yang<sup>1,2,3†</sup>, Xiuhui Shi<sup>4†</sup>, Wei Ba<sup>10</sup>, Zhigang Song<sup>10</sup>, Haijing Luan<sup>6</sup>, Taiyuan Hu<sup>6</sup>, Senlin Lin<sup>7</sup>, Jiguang Wang<sup>2,3\*</sup>, Shaohua Kevin Zhou<sup>1,5,8,9\*</sup>, Rui Yan<sup>1,5\*</sup>

<sup>1</sup>School of Biomedical Engineering, Division of Life Sciences and Medicine, University of Science and Technology of China, Hefei, Anhui, China.

<sup>2</sup>Division of Life Science, Department of Chemical and Biological Engineering, State Key Laboratory of Nervous System Disorders, The Hong Kong University of Science and Technology, Hong Kong SAR, China. <sup>3</sup>SIAT-HKUST Joint Laboratory of Cell Evolution and Digital Health, HKUST Shenzhen-Hong Kong Collaborative Innovation Research Institute, Futian, Shenzhen, China.

<sup>4</sup>Department of Hepatobiliary Surgery, The First Afiliated Hospital of USTC, Division of Life Sciences and Medicine, University of Science and Technology of China, Hefei, Anhui, China.

<sup>5</sup>Center for Medical Imaging, Robotics, Analytic Computing & Learning (MIRACLE), Suzhou Institute for Advanced Research, USTC, Suzhou, Jiangsu, China.

<sup>6</sup>Computer Network Information Center, Chinese Academy of Sciences, Beijing, China. <sup>7</sup>Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China. <sup>8</sup>Jiangsu Provincial Key Laboratory of Multimodal Digital Twin Technology, Suzhou, Jiangsu, China. <sup>9</sup>Key Laboratory of Precision and Intelligent Chemistry, USTC, Hefei, Anhui, China. <sup>10</sup>Department of Pathology, Chinese PLA General Hospital, Beijing, China.

\*Corresponding author(s). E-mail(s): jgwang@ust.hk; skevinzhou@ustc.edu.cn; yanrui@ustc.edu.cn; <sup>†</sup>These authors contributed equally to this work.

## Abstract

Whole slide image (WSI) analysis has emerged as an increasingly essential technique in computational pathology. Recent advances in the pathology foundation models (FMs) have demonstrated significant advantages in deriving meaningful patch-level or slide-level multi-scale features from WSIs. However, current pathology FMs have exhibited substantial heterogeneity caused by diverse private training datasets and diferent network architectures. This heterogeneity introduces performance variability when we utilize the features from diferent FMs in the downstream tasks. To fully explore the advantages of multiple FMs efectively, in this work, we propose a novel framework for the fusion o multi-scale heterogeneous pathology FMs, called FuseCPath, yielding a model with a superior ensemble performance The main contributions of our framework can be summarized as follows: (i) Tc guarantee the representativeness of the training patches, we propose a multi-view clustering-based method to filter out the discriminative patches via multiple FMs’ embeddings. (ii) To efectively fuse the patch-level FMs, we devise a cluster-level re-embedding strategy to online capture patch-level local features. (iii) To efectively fuse the slide-level FMs, we devise a collaborative distillation strategy to explore the connections between slide-level FMs. Extensive experiments demonstrate that the proposed FuseCPath achieves state-of-the-art performance across multiple tasks on diverse datasets.

Keywords: Foundation model, Histopathological image analysis, Multi-model integration, Information fusion

## 1 Introduction

Pathological diagnosis is the gold standard for cancer diagnosis, while whole slide image (WSI) analysis occupies a core position in computational pathology (CPath) and can support key tasks such as cancer subtyping [1, 2], survival prediction [3–5], and biomarker prediction [6–8]. In recent years, the rapid development of pathology foundation models (FMs) [9–12] has brought about a revolutionary transformation in this field.

Current pathology FMs can be categorized into two distinct types at diferent scales, which are patch-level FMs [9, 10, 13] and slide-level FMs [11, 12, 14, 15]. The patch-level FMs are trained with the tiled patches of WSIs. The patch embeddings derived from patch-level FMs will be aggregated with Multiple-Instance Learning (MIL) for the training of downstream tasks in CPath. Most of the patch-level FMs are trained with self-supervised learning methods of diferent architectures (e.g., Dino-v2 [16] or MAE [17]), using diferent private datasets. Diferent from patch-level FMs, the slide-level FMs are capable of constructing slide embeddings with unsupervised learning. Similar to patch-level FMs, the architectures of backbone models and training datasets difer significantly in each slide-level FM. In conclusion, we define these FMs diferences as heterogeneity in the pathology FMs.

Because of the heterogeneity, the performance in diferent downstream tasks and the learned tissue morphologies are diverse across diferent FMs [18]. To ensure the performance of FMs on downstream tasks, the most trivial strategy is to select a foundation model with the best performance on the corresponding downstream task, as shown in Figure 1(a). However, this strategy contains obvious shortcomings. Firstly, re-training a foundation model with our own training datasets may not reproduce the optimal performance. Secondly, when we are facing more than one downstream task, re-training many FMs simultaneously is not a flexible solution. Consequently, based on the concept of ensemble learning, it is an efective way to fuse the patch-level and slide-level embeddings from the heterogeneous FMs into a single proxy model for training, as shown in Figure 1(b). By combining the strengths of each individual foundation model, we will obtain a fused model with improved performance on downstream tasks [18]. However, there still exist two major challenges hindering the fusion of heterogeneous FMs. Firstly, the heterogeneity in the pathology FMs contributes to diverse dimensions and information of the embeddings. It is essential to comprehensively capture the connections between the patchlevel or slide-level embeddings derived from heterogeneous FMs. Secondly, the scale gaps between patch-level and slide-level embeddings. The representation information captured by patch-level and slide-level embeddings is distributed at diferent scales. We need to fully leverage the representational information from slide-level embeddings to assist in training models with patch-level embeddings.

To address these challenges, in this work, we propose a novel framework called FuseCPath for the fusion of multi-scale (patch-level and slidelevel) heterogeneous pathology FMs. Figure 1(c) illustrates the key ideas and contributions of the proposed FuseCPath. The fusion of patch-level FMs aims at fusing the patch-level local features from diverse FMs (Figure 1(c)). First, we propose a multi-view clustering strategy to select representative patches utilizing the meaningful features captured from heterogeneous patch-level FMs. Second, we devise a cluster-level feature reembedding transformer to discover the relations between patch-level FMs in feature space. The fusion of slide-level FMs aims at fusing the slidelevel global features from diverse FMs (Figure 1(c)). Consequently, we devise a collaborative distillation module to efectively utilize the representative global features residing in slide embeddings as a teacher model. Equipped with the above modules, the FuseCPath framework will be capable of fusing the heterogeneous FMs efectively. The code is publicly available from https://github. com/ZhidongYang/FuseCPath.

• We propose FuseCPath, a novel framework for fusing the heterogeneous multi-scale (patchlevel and slide-level) pathology FMs to integrate a model with better performance.

• We devise a novel online feature re-embedding transformer that operates on filtered discriminative patch-level features with multi-view clustering. The proposed online re-embedding efectively addresses the issue of fusing the heterogeneous patch-level FMs by capturing meaningful features locally and connecting the patch embeddings across diverse patch-level FMs.

![](images/a21fc1d82e981684c42cee9219d5926578ba413bdb98208fb3d507131ce83872.jpg)  
Fig. 1 (a) Conventional foundation model-based WSI analysis paradigm. To achieve optimal performance on downstream tasks, the most straightforward strategy is to select a patch-level or slide-level foundation model that exhibits the strongest performance on the target task. (b) Proposed heterogeneous foundation model fusion-based WSI analysis paradigm. Based on the concept of ensemble learning, a framework for the fusion of multi-scale (patch-level and slide-level) heterogeneous pathology FMs will yield a model with superior performance. (c) The key ideas and contributions of this article.

• We propose a novel collaborative distillation module for the fusion of slide-level FMs that systematically bridges the scale gap between the fusion of patch-level and slide-level FMs. The slide-level FMs will serve as teacher models to provide global representations of WSIs.

• Extensive experiments demonstrate that the FuseCPath will ensemble a new model with superior performance across various downstream tasks in WSI analysis.

## 2 Related work

## 2.1 Pathology foundation model

Recent advances in pathology FMs have employed diverse architectural and training paradigms, predominantly utilizing self-supervised learning (SSL) techniques [19, 20] to extract meaningful representations from unannotated patches in WSIs. SSL-based approaches can be summarized as follows: (1) contrastive learning frameworks such as REMEDIS [21], which adapt SimCLR frameworks [20] by maximizing feature similarity between comparable regions within individual WSIs while minimizing the similarity across disparate slide regions; (2) masked image modeling adopted by Prov-gigapath [14], CONCH [9], and BEPH [22], where random patch occlusion forces models to reconstruct masked tissue patterns, thereby capturing robust contextual relationships; and (3) knowledge distillation implementations exemplified by Virchow2 [10], UNI [13], and Hibou [23], which employ DINO-based frameworks [16] to distill the knowledge from teacher to student models, yielding compact yet generalizable representations without extensive annotations.

Slide-level representation learning has emerged as an essential approach for generating taskagnostic embeddings through unsupervised learning. Pioneering work by Chen et al. [24] proposed the HIPT method by devising a hierarchical self-distillation for WSI-level representation learning. Lazard et al. [25] developed a contrastive learning-based framework using augmented patch ensembles. Subsequent innovations include ProvgigaPath (SE)’s masked autoencoder architecture [14] for generating slide representations and several multi-modal-based pretraining FMs [11, 15]. Existing slide-level FMs universally require substantial training data (over 10K WSIs) [12, 24, 25], while PRISM [11] and GigaPath-SE (slide encoder) [14] utilize more WSIs. With the rapid development of multi-omics techniques, FMs will be capable of bridging the H&E-stained pathology images to other omics data [26–28].

## 2.2 Multiple instance learning in WSI analysis

Multi-instance learning (MIL) is a predominant weakly supervised learning strategy widely adopted in the applications of downstream tasks for WSI analysis [29–31], which solves the problem of lacking high-quality annotations. The attention-based deep multi-instance learning (AB-MIL) proposed by [32] first adopts convolutional neural networks (CNNs) to multi-instance learning. This technique is widely extended to the application for WSI image analysis. [29] introduced a recalibrated multi-instance learning framework (RMDL) for the classification of whole slide images (WSIs) of gastric tissues. The RMDL employs a convolutional neural network (CNN) to identify discriminative instances (the image patches) within each WSI and subsequently trains the model exclusively on these selected instances. RMDL captures dependencies among instances and dynamically recalibrates their features based on the coeficients derived from fused feature representations. [31] developed a dual-stream multiple instance learning network (DSMIL) comprising two synergistic streams: one learns an instance-level classifier using max-pooling to identify the highest-scoring (critical) instance, while the other computes attention scores for instances based on their proximity to the critical instance.

[33] proposed DeepAttnMISL, a survival pre diction model that integrates attention mecha nisms with multi-instance learning. This approach clusters the patches extracted from WSIs into phenotypically distinct groups. Then it selects representative patches from each cluster and pro cesses them through a Siamese multi-instance fully convolutional network. The model subse quently aggregates features via attention-based multiple instance learning (AB-MIL) pooling to predict patient survival risk. Similarly, [30] proposed CLAM, which also operates in two stages: first, patches are encoded into feature vectors using a pre-trained CNN, and then these fea tures are processed by a clustering-constrained attention mechanism within a multiple instance learning framework to produce final predictions. [34] developed DeepSMILE, a two-stage frame work wherein the first stage employs the con trastive learning method SimCLR for patch-level feature extraction, generating representative fea ture embeddings. The second stage incorporates these features into the proposed VarMIL, which is an extension of AB-MIL that introduces a fea ture variability module to explicitly model tumor heterogeneity. Yan et al. [6] proposed a hierarchi cal deep multi-instance learning-based framework called HD-MIL to accurately predict gene muta tions in bladder cancer by leveraging a contrastive learning framework called Bootstrap Your Own Latent (BYOL) to derive high-quality feature representations. Yang et al. [35] proposed to incorporate the Selective Scan Space State Sequential Model (Mamba) in Multiple Instance Learning (MIL) for long sequence modeling with linear complexity to adjust the high-resolution of WSIs. Similarly, Tang et al. [36] proposed a re-embedding strategy called $\mathrm { R ^ { 2 } T }$ to online captures foundation model-level local features and establishes connections across diferent regions. Additionally, the proposed $\mathrm { R ^ { 2 } T }$ can be integrated into MIL models $\mathrm { \bar { ( R ^ { 2 } T - M I L ) } }$ to improve the performance of several downstream tasks.

## 2.3 Patch selection in WSI analysis

Due to the gigapixel-scale high resolution of WSIs, it is challenging to fit WSIs to the GPU devices in an end-to-end manner. One efective solution is to crop the images into patches for training. Several approaches are proposed to implement this module. [37] first proposed DeepGraph-Surv, a survival analysis model that employs graph convolutional networks (GCNs) by randomly sampling over 1,000 patches from each WSI to construct graphs for classification, achieving C-indices of 0.66 and 0.62 on TCGA-LUSC and TCGA-GBM datasets, respectively. [38] proposed an integrated framework combining graph neural networks with attention-based multiple instance learning for colorectal cancer TNM staging, where they extracted texture features from randomly selected patches, constructed graphs from these patches, and used them as instances in their classification model. While demonstrating broad applicability and straightforward implementation, this approach may be limited by the potential lack of representativeness in randomly sampled patches, which could impact classification performance.

To ensure the representativeness of patches, the strategy of approximating Regions of interest (RoI) is adopted. The RoI can be approximated using several distinctive patches, and several notable methods have been developed based on this conclusion ([39–41]). For instance, [40] employed the color-based strategy outlined in Yottixel ([42]) to extract several patches from WSIs, and these patches are modeled by a fully connected graph. In this way, the task of classifying WSIs is converted into graph classification. In [40], the authors gathered 1,026 WSIs from the TCGA lung cancer dataset, achieving an accuracy of 88.8. [41] utilized weakly supervised learning to categorize lung cancer into four subtypes. This method first utilizes a patch-based full convolutional neural network to identify distinctive blocks, then applies diferent block selection and feature aggregation strategies based on probability maps to generate a global representation for the WSI. Finally, these global representations will serve as input to a random forest, which will produce the classification results.

The clustering strategy is also an efective way to provide prior knowledge for patch selection. Based on the result of the feature clustering, several clustering-based methods ([6, 43, 44]) are proposed to guarantee the representativeness of the selected patches. The survival prediction method (WSISA) developed by [43] can make efective use of all distinguishing patch features in WSIs, thereby significantly enhancing survival prediction performance compared to existing methods. WSISA first selects hundreds of patches from each WSI and then further clusters these selected patches. Then, it selects clusters based on the patch-level prediction performance using CNN, and fuses them to make the final prediction. Inspired by this method, [44] firstly combines the advantages of selecting patches in Regionsof-Interest (RoI) from detected cancer areas and clusters, using the data from 253 bladder cancer patients in the TCGA dataset. The method proposed by [6] proposed to select representative patches from clustered detected cancer areas using high-quality embeddings derived from a BYOL-based pre-trained model.

## 3 Method

FuseCPath is a framework for the fusion of multi-scale (patch-level and slide-level) heterogeneous pathology FMs, which contributes to a significant performance improvement on the WSI image analysis across multiple tasks. We provide a brief overview of FuseCPath below, and Figure $2 ( \mathrm { a } )$ presents more details. Given a set of WSIs $\{ \dot { X _ { i } } | \dot { X _ { i } } \in \mathbb { R } ^ { d _ { x } \times d _ { y } \times 3 } \}$ , FuseCPath will simultaneously fuse the patch embeddings and slide embeddings of $X _ { i } .$ . For the fusion of patch-level FMs, the FuseCPath will first cluster the patch embeddings with multiview spectral clustering to find representative patches. Then, a Clusterlevel Re-embedding Transformer $\mathrm { ( C R ^ { 2 } T ) }$ is used to online fuse the patch embeddings, and the Attention-Based Multiple instance learning (AB-MIL) to aggregate the re-embedded features. For the fusion of slide-level FMs, FuseCPath regards the slide-level embeddings as the teacher models’ information, which is implemented with a collaborative distillation during training.

## 3.1 Multi-view patch features clustering

Due to the extremely high resolution of the WSIs, it will be a challenging operation to input all the patches extracted from the WSIs for training. Consequently, a typical solution is to select a subset of the patches randomly with a fixed amount of patches. However, the random selection can not guarantee the representativeness of the patches for training. In this work, we devise a cluster-based strategy to select representative patch embeddings from multiple FMs for training.

(b) Patch-level features re-embedding Multi-view features clustering  
![](images/7926f58c56a512ec101d32458c4e2ae4570840d6c7a217f78146153ce22f89a7.jpg)

![](images/8d5a48176334e1f59f68186df8b3a00789356603541c9812ac1b7792ae9a96d6.jpg)  
Fig. 2 The overall architecture and main components of the proposed FuseCPath framework. (a) The overall architecture of the FuseCPath framework. The FuseCPath can be divided into two essential branches, which are patch-level features reembedding and slide-level features collaborative distillation. (b) The demonstration of patch-level features re-embedding. Representative features can be summarized with multi-view clustering and online re-embedding. (c) The re-embedded features aggregation module is implemented by AB-MIL.

![](images/863351048c72511e63390d1b3c1b419e6484045c5bce50f6693f96bd9dc28e5a.jpg)  
Fig. 3 Details of multi-view spectral clustering (MVSC). The patch embeddings from diverse patch-level FMs can be regarded as a view of the original WSI dataset.

Firstly, the patch embeddings are derived from pre-trained heterogeneous patch-level FMs.

$$
\mathbf {H} ^ {f _ {p e} ^ {i}} = f _ {p e} ^ {i} (X), \mathbf {H} ^ {f _ {p e} ^ {i}} \in \mathbb {R} ^ {N _ {X} \times d _ {p e} ^ {i}},\tag{1}
$$

where $f _ { p e } ^ { i }$ denotes the patch-level foundation model, and $f _ { p e } ^ { i } \in \mathcal { F } _ { p e } = \{ \mathrm { C O N C H }$ , Virchow2, Gigapath}. $\mathbf { H } ^ { f _ { p e } ^ { i } }$ denotes the patch embeddings derived from the foundation model $\mathbf { H } ^ { f _ { p e } ^ { i } }$ $N _ { X }$ denotes the complete number of patches extracted from WSI X. $d _ { p e } ^ { i }$ denotes the dimension of the embeddings extracted from $f _ { p e } ^ { i }$ , where $d _ { p e } ^ { i } \in$ $\mathcal { D } _ { p e } = \{ 7 6 8 , 2 5 6 0 , 1 5 3 6 \}$

Since we need to simultaneously consider the representativeness of the selected patches based on the patch embeddings from multiple FMs $f _ { p e } ^ { i } ,$ the traditional K-means cluster method is not suitable in this situation. Thus, the multi-view spectral clustering is selected as an optimal solution. The patch embeddings $\mathbf { H } ^ { f _ { p e } ^ { i } }$ derived from a distinct foundation model can be regarded as a view of the original WSI, and each view provides a diverse representation of the WSI.

$$
\mathcal {H} ^ {f _ {p e}} = \{\mathbf {H} ^ {f _ {p e} ^ {1}}, \dots , \mathbf {H} ^ {f _ {p e} ^ {n}} \}.\tag{2}
$$

where the extended tensor $\mathcal { H } ^ { f _ { p e } }$ is the input of multi-view spectral clustering. Consequently, the patches will be clustered into K clusters, and then $N _ { K }$ patches will be selected from the clusters. As a result, $N _ { K } \times K$ patches will be selected as the representative patch for training embeddings. Figure

3 illustrates the main process of multi-view clustering for heterogeneous patch embeddings from multiple patch-level FMs.

## 3.2 Patch-level features re-embedding

With the selected patch embeddings from the foundation model, existing methods chose to finetune the original model using the obtained features to adjust the downstream task. However, when the patch embeddings are derived from FMs with heterogeneous architectures, the models will be fine-tuned separately with our training data. From the perspective of MIL, this process can be formulated as follows.

$$
z = \mathcal {A} (f _ {p e} ^ {1} (X),..., f _ {p e} ^ {n} (X)),\tag{3}
$$

where z denotes the aggregated slide-level features using patch-level features from multiple sources of FMs $f _ { p e } ^ { i } . \ A ( \cdot )$ denotes the mapping function of feature aggregation. The performance of this strategy is limited by the diference between our own fine-tuning datasets and the original training datasets for the FMs. A more efective solution is based on online simultaneous training using a consistent training paradigm. Thus, we devise an online cluster-level patch features re-embedding strategy to fuse the embeddings from heterogeneous FMs into a single model. This strategy can be formulated as follows.

$$
z = \mathcal {A} (\mathcal {R} (f _ {p e} ^ {1} (X),..., f _ {p e} ^ {n} (X))),\tag{4}
$$

where $\mathcal { R } ( \cdot )$ denotes the mapping function of online features re-embedding. Inspired by R<sup>2</sup>Transformer in [36], we opt for the Regional Multi-head Selfattention (R-MSA) and Cross Regional Multihead Self-Attention (CR-MSA) as the base model for our strategy. Figure 2(b) demonstrates the procedure of online re-embedding for patch-level features in our FuseCPath.

The number of patches in a high-resolution WSI is too large to serve as the input of the Transformer-based models. Especially in the situation of fusing the embeddings from multiple FMs, the embeddings are equipped with much higher dimensions. Thus, we need to avoid the Out-of-Memory issue. The R-MSA [36] strategy addresses this issue by partitioning the patches into independent regions, where multi-head selfattention is performed on these regions. In this work, the solution to this issue goes one step further. Unlike the vanilla R-MSA, the input embeddings for our re-embedding module have been summarized by multi-view clustering. Our R-MSA only needs to focus on the sparsely selected patches from the clusters, which are representative enough for training. Hence, this strategy can be refactored to Clustered Multi-head Selfattention (C-MSA). Previous work [3] has proved the representativeness of the patches selected from the clusters. The C-MSA can be formulated as follows.

$$
\begin{array}{r l} & {\mathbf {C} _ {1},..., \mathbf {C} _ {K} = \mathrm{Cluster} (\mathbf {H} ^ {f _ {p e} ^ {1}},..., \mathbf {H} ^ {f _ {p e} ^ {n}}),} \\ & {\mathbf {H} _ {s} ^ {f _ {p e} ^ {1}},..., \mathbf {H} _ {s} ^ {f _ {p e} ^ {N}} = \mathrm{Selection} (\mathbf {C} _ {1},..., \mathbf {C} _ {K}),} \\ & {\mathbf {H} _ {s} ^ {f _ {p e} ^ {i}} \in \mathbb {R} ^ {(N _ {K} \times K) \times d _ {p e} ^ {i}},} \\ & {\hat {\mathbf {Z}} _ {m} ^ {p e} = \mathrm{MSA} (\mathrm{LN} ([ \mathbf {H} _ {s} ^ {f _ {p e} ^ {1}},..., \mathbf {H} _ {s} ^ {f _ {p e} ^ {N}} ])) + [ \mathbf {H} _ {s} ^ {f _ {p e} ^ {1}},..., \mathbf {H} _ {s} ^ {f _ {p e} ^ {N}} ],} \\ & {\hat {\mathbf {Z}} _ {m} ^ {p e} \in \mathbb {R} ^ {(N _ {K} \times K) \times D},} \end{array}\tag{5}
$$

where $\mathbf { C } _ { 1 } , . . . , \mathbf { C } _ { K }$ denote the clusters. $\mathbf { H } _ { s } ^ { f _ { p e } ^ { 1 } } , . . . , \mathbf { H } _ { s } ^ { f _ { p e } ^ { N } }$ denote the selected patch embeddings from the clusters. $\hat { \mathbf { Z } } _ { m } ^ { p e }$ denotes the encoded embeddings $\left( D \ = \ \Sigma _ { i } d _ { p e } ^ { i } \right)$ . We adopt the Position Encoding Generator (PEG) implemented using a 1-D convolutional layer to the encoded embeddings $\hat { \mathbf { Z } } _ { m } ^ { p e }$

$$
\alpha_ {i j} = \mathrm{SoftMax} (\mathbf {e} _ {i j} + \mathrm{PEG} (\mathbf {e} _ {i j})),\tag{6}
$$

where $\alpha _ { i j }$ is the attention weights of $\hat { \mathbf { Z } } _ { m j } ^ { p e }$ with respect to $\hat { \mathbf { Z } } _ { m i } ^ { p e } . ~ \mathbf { e } _ { i j }$ is a tensor calculated with a scaled dot-product attention using $\hat { \mathbf { Z } } _ { m } ^ { p e }$ [36].

Similar to Cross-regional Multi-Head Self-Attention (CR-MSA), it is essential to consider the semantic context between the selected patches for the downstream tasks in WSI analysis. Therefore, we need to model the connections between cluster-level patches using CR-MSA, which should be referred to as Cross-cluster Multi-Head Self-Attention (CC-MSA) in our work. The cluster-level features will be fused with the vanilla MSA module and normalized by the MinMax(·)

function. This process can be formulated as follows.

$$
\begin{array}{r l} & {\mathbf {R} _ {a} ^ {p e} = \mathrm{SoftMax} _ {k = 1} ^ {K} (\hat {\mathbf {Z}} _ {m k} ^ {p e} \Phi) ^ {T} \hat {\mathbf {Z}} _ {m} ^ {p e},} \\ & {\mathbf {W} _ {d} ^ {p e} = \mathrm{MinMax} _ {k = 1} ^ {K} (\hat {\mathbf {Z}} _ {m k} ^ {p e} \Phi),} \\ & {\hat {\mathbf {W}} _ {d} ^ {p e} = \mathrm{SoftMax} _ {g = 1} ^ {G} (\hat {\mathbf {Z}} _ {m g} ^ {p e} \Phi) \in \mathbb {R} ^ {G \times 1},} \\ & {\hat {\mathbf {Z}} ^ {p e} = (\mathbf {W} _ {d} ^ {p e}) ^ {T} \mathrm{MSA} (\mathbf {R} _ {a} ^ {p e}) \hat {\mathbf {W}} _ {d} ^ {p e}.} \end{array}\tag{7}
$$

where $\Phi \in \mathbb { R } ^ { D \times G }$ denotes learnable parameters, $\mathbf { W } _ { d } ^ { p e }$ denotes the normalization weights for the fused patch-level features MSA(R<sup>pe</sup>). The CC-MSA is calculated at the cluster level using patch embeddings.

## 3.3 Re-embedded features aggregation

Re-embedded features aggregation is an essential module of our heterogeneous FMs fusion framework. The multi-instance learning (MIL) is widely adopted as an efective solution in WSI analysis, where the labels y are assigned to each WSI. And the WSI X can be defined as bag, the patches within X are instances. To efectively demonstrate the features aggregation, we will first briefly introduce MIL and then proceed to the features aggregation in our FuseCPath.

The multi-instance learning (MIL) is a useful weakly supervised learning method in WSI analysis. The formulation of MIL is shown as follows. The dataset consists of bags assigned with labels $\mathbf { y } = \{ y _ { 1 } , y _ { 2 } , . . . , y _ { D } \}$ . Each bag contains several instances. If at least one instance in a bag is positive, the bag will be considered positive, and if all instances in a bag are negative, the bag will be considered negative. Take the situation in binary classification as an example, we define $B \ = \ \{ ( x _ { 1 } , y _ { 1 } ) , . . . , ( x _ { D } , y _ { D } ) \}$ as a bag. $x _ { d } ~ \left( d ~ \in \right.$ $\{ 1 , 2 , . . . , D \} )$ are instances of $B ,$ , each with labels $y _ { d } \in 0 , 1$ . Consequently, the label $Y$ of B is given by:

$$
Y = \prod_ {y _ {d} \in {\bf y}} (y _ {d}) = \left\{ \begin{array}{l l} 0, \forall y _ {d} = 0, \\ 1, \exists y _ {d} = 1. \end{array} \right.\tag{8}
$$

In this work, the features aggregation $\boldsymbol { \mathcal { A } } ( \cdot )$ is implemented by Attention-based multi-instances learning (AB-MIL), which integrates the strengths of attention-based MIL pooling for aggregating the features $\hat { \mathbf { Z } } ^ { p e }$ into a single feature vector ${ \bf Z } ^ { p e }$ with a weighted averaging operation. Figure $2 ( \mathrm { c ) }$ demonstrates the procedure of feature aggregation via AB-MIL. In this work, the input of AB-MIL is the re-embedded patch-level FMs features $\hat { \mathbf { Z } } ^ { p e } \in \mathbb { R } ^ { ( N _ { K } \times K ) \times D }$ . By adopting AB-MIL as pooling module, the aggregated feature ${ \bf Z } ^ { p e }$ can be formulated as follows:

$$
\begin{array}{l} \mathbf {Z} ^ {p e} = \mathcal {A} (\hat {\mathbf {Z}} ^ {p e}) = \sum_ {d = 1} ^ {D} (a _ {d} (\hat {\mathbf {Z}} _ {d} ^ {p e}) \cdot \hat {\mathbf {Z}} _ {d} ^ {p e}), \\ a _ {d} (\hat {\mathbf {Z}} _ {d} ^ {p e}) = \frac {\exp \left(\mathbf {W} ^ {T} \tanh \left(\mathbf {V} (\hat {\mathbf {Z}} _ {d} ^ {p e}) ^ {T}\right)\right)}{\sum_ {j = 1} ^ {D} \exp \left(\mathbf {W} ^ {T} \tanh \left(\mathbf {V} (\hat {\mathbf {Z}} _ {j} ^ {p e}) ^ {T}\right)\right)}. \end{array}\tag{9}
$$

where $a _ { d } ( \cdot )$ denotes the attention operation corresponding to the embedding $\hat { \mathbf { Z } } _ { d } ^ { p e } , \hat { \mathbf { W } } \in \mathbb { R } ^ { ( N _ { K } \times K ) \times 1 }$ and $V ~ \in ~ \mathbb { R } ^ { ( N _ { K } \times K ) \times D }$ are learnable parameters. The aggregated feature ${ \bf Z } ^ { p e }$ will be the input of the slide-level collaborative distillation module, which is an essential step to fuse the slide-level FMs.

## 3.4 Slide-level collaborative distillation

Slide-level foundation model is capable of yielding a high-level representation of WSI, which is more concise for a downstream task fine-tuning. However, it is a problem that fuse these slide-level global representations with patch-level local representations. This is challenging due to the dimensional gaps. In the proposed FuseCPath, we try to solve this problem by regarding the slide-level features as soft labels derived from teacher models. Consequently, we propose a slide-level collaborative distillation strategy to fuse slide-level FMs that contain global information simultaneously.

Consider the re-embedded patch-level features ${ \bf Z } ^ { p e }$ and slide-level features $\bar { \mathbf { L } _ { s e } ^ { 1 } } , . . . , \mathbf { L } _ { s e } ^ { n }$ derived from N heterogeneous slide-level FMs $F _ { s e } ^ { \mathrm { 1 } } , . . . , F _ { s e } ^ { N }$ each slide-level FM is regarded as a teacher model. We first project the embeddings $\mathbf { L } _ { s e } ^ { 1 } , . . . , \mathbf { L } _ { s e } ^ { n }$ with a linear layer to ensure the same dimensions of the features.

$$
\mathbf {h} _ {s e} ^ {i} = \mathrm{Linear} _ {i} (\mathbf {L} _ {s e} ^ {i}), \mathbf {L} _ {s e} ^ {i} \in \mathbb {R} ^ {1 \times d _ {s e} ^ {i}},\tag{10}
$$

where $\mathbf { h } _ { s e } ^ { i }$ denotes the projected features subject to the teacher model $\begin{array} { r l r l } { F _ { s e } ^ { i } } & { { } \in } & { \mathcal { F } _ { s e } } & { { } = } \end{array}$ {Gigapath-SE, TITAN, PRISM}. $d _ { s e } ^ { i }$ denotes the dimension of slide-level embeddings, where $d _ { s e } ^ { i } \in$ $\mathcal { D } _ { s e } = \{ 7 6 8 , 1 2 8 0 \}$ $\mathbf { h } _ { s e } ^ { i }$ is usually calculated by a Linear layer. Similarly, the patch-level FMs are regarded as a student model. The projection layer is formulated as follows.

$$
\mathbf {h} _ {p e} = \mathrm{Linear} (\mathbf {Z} _ {p e}),\tag{11}
$$

To ensure performance, the features will be softened by temperature τ using the softmax function. In this work, the temperature τ is set to 3 for the classification task, and τ is set to 1 for the regression task.

$$
\bar {\mathbf {h}} _ {s e} ^ {i} = \operatorname{SoftMax} (\frac {\mathbf {h} _ {s e} ^ {i}}{\tau}), \bar {\mathbf {h}} _ {p e} = \operatorname{SoftMax} (\frac {\mathbf {h} _ {p e}}{\tau}),\tag{12}
$$

With the softened distribution $\bar { \mathbf { h } } _ { s e } ^ { i }$ and $\bar { \mathbf { h } } _ { p e }$ , the Kullback-Leibler Divergence KL(·∥·) is adopted to formulate the distillation loss function $\mathcal { L } _ { d i s t } ^ { i }$ of the teacher model (slide-level FM), which is formulated as follows.

$$
\begin{array}{r} \mathcal {L} _ {d i s t} ^ {i} = \tau^ {2} \cdot \mathrm{KL} (\bar {\mathbf {h}} _ {p e} \| \bar {\mathbf {h}} _ {s e} ^ {i}), \\ = \tau^ {2} \cdot \sum_ {c = 1} ^ {C} \bar {h} _ {p e} ^ {c} \log \frac {\bar {h} _ {p e} ^ {c}}{\bar {h} _ {s e} ^ {c}}, \end{array}\tag{13}
$$

where $C$ denotes the dimension of the softened distributions. Given label $\mathbf { y } ,$ the combined loss function of distillation is formulated as follows:

$$
\mathcal {L} _ {f u s e} = \lambda \mathcal {L} _ {t a s k} (\mathbf {h} _ {p e}, \mathbf {y}) + (1 - \lambda) \frac {1}{N} \sum_ {i = 1} ^ {N} \mathcal {L} _ {d i s t} ^ {i}.\tag{14}
$$

where $\mathcal { L } _ { t a s k }$ is related to the downstream tasks. The $\mathcal { L } _ { t a s k }$ for biomarker prediction will be a binary cross-entropy loss function (mutation refers to 1, and wild-type refers to 0). For the prediction of gene expression, the task will be modeled as a regression problem. So $\mathcal { L } _ { t a s k }$ will be the Mean Squared Error (MSE). λ denotes the weight for balancing the student and teacher models. For survival analysis, the $\mathcal { L } _ { t a s k }$ will be formulated using the Cox proportional hazard model [3]. We summarize the main training procedure for our FuseCPath framework in Algorithm 1.

## 3.5 Implementation details

The complete procedure of our FuseCPath framework consists of the following essential modules: multi-view patch features clustering, patch-level features re-embedding, features aggregation, and slide-level distillation. All experiments in this paper were finished on four NVIDIA A100 80G GPUs with an Ubuntu 20.04 system. The implementation of FuseCPath is mainly based on the Pytorch framework, Trident [45], OpenSlide, and Scikit-Learn packages.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1: Procedure of FuseCPath

Input: Whole Slide Image X;
Task-related label y.

Output: Fused features  $Z^{f}$ ;
Prediction result  $\hat{y}$ .

1 Procedure FuseCPath(X, y):

2  $H^{f_{pe}^{1}}, ..., H^{f_{pe}^{n}} \leftarrow f_{pe}^{1}(X), ..., f_{pe}^{n}(X);$ $L^{f_{se}^{1}}, ..., L^{f_{se}^{n}} \leftarrow F_{se}^{1}(X), ..., F_{se}^{N}(X);$ $C_{1}, ..., C_{K} \leftarrow Cluster(H^{f_{pe}^{1}}, ..., H^{f_{pe}^{n}});$ $H_{s}^{f_{pe}^{1}}, ..., H_{s}^{f_{pe}^{N}} \leftarrow$ 

Selection( $C_{1}, ..., C_{K}$ );

 $\hat{Z}^{pe} \leftarrow ReEmbedding(H_{s}^{f_{pe}^{1}}, ..., H_{s}^{f_{pe}^{N}});$ $Z^{pe} \leftarrow$ 

ReEmbeddedFeaturesAggregation( $\hat{Z}^{pe}$ );

 $Z^{f} \leftarrow Distillation(Z^{pe}; L^{f_{se}^{1}}, ..., L^{f_{se}^{n}});$ $\hat{y} \leftarrow TaskHeader(Z^{f}; y);$ 

return  $Z^{f}, \hat{y};$
</div>

Multi-view patch features clustering. In the cluster module, the implementation mainly relies on the mvlearn and Trident packages. The input WSIs will first be fed into the Deeplabv3 model to extract the tissue regions. Then the patches are tiled from the tissue regions, and all the patches are sized by 256×256. Patch embeddings are derived with patch-level FMs ${ \mathcal { F } } _ { p e } ~ =$ {CONCH, Virchow2, Gigapath} using these tiled patches. Finally, the embeddings derived from distinct FMs will be concatenated into a list with the corresponding indices, which is the input of multi-view spectral clustering. The patches will be clustered into K = 50 clusters.

Patch-level features re-embedding. In the re-embedding module, the implementation mainly relies on the $\mathrm { \bar { R } ^ { 2 } T }$ and Trident packages. We adopt hierarchical sparse self-attention $\left( \mathrm { t o p } { - } k = 8 \right)$ to accelerate the training process and meanwhile suppress the over-fitting problem [3]. The input embeddings for training are selected from the clustered patch embeddings. We select $N _ { K } ~ = ~ 1 0$ patches from each cluster and a total of 500 patches for each WSI. The re-embedding contains two layers of C-MSA and one layer CC-MSA with 10% dropout during training. For the detailed parameters for the re-embedding module, the batch size is 64. The learning rate starts with 1e-4. The model is optimized using Stochastic Gradient Descent (SGD), where the momentum parameter is $m = 0 . 9$ and the learning rate decay ratio is 5e-5.

Slide-level distillation. In the slide-level distillation module, the implementation mainly relies on the PyTorch and Trident packages. We derive the slide embeddings from the slide-level FMs Gigapath-SE, TITAN, and PRISM for the soft labels during training. Each Linear (·) in Equation 10 is implemented by a linear projection layer to align with the re-embedded patch-level features. To balance the weight for teacher models, we set $\lambda = 0 . 5$ and N = 3 in Equation 14 during training.

The repeated selection-based data augmentation plays a critical role in enhancing FuseCPath’s performance. For 5-fold cross-validation, we partition all WSIs into training and validation sets with a ratio of 80%:20%. The repeated summarization process is applied separately to each partitioned dataset as follows: For each of W WSIs, we first perform clustering to generate $K = 5 0$ clusters, then randomly select ${ K _ { N } \mathrm { ~ \Omega ~ } = \mathrm { ~ \Omega ~ } ^ { 1 0 } }$ patches from each cluster. This operation will generate 500 representative patches per WSI. By repeating this procedure $N _ { R } = 5 0$ times, we obtain $N _ { R } = 5 0$ distinct summarizations for each WSI, efectively expanding the dataset size from W to $W \times N _ { R }$ . To address the remaining class imbalance, we apply conventional augmentation techniques, including random flipping, cropping, rotation, scaling, and blurring, to enhance the training dataset quality. The fused embeddings will be input to the Multi-layer perceptrons (MLPs) for prediction or regression tasks.

## 4 Experiment

## 4.1 Dataset description and evaluation metrics

To evaluate and compare the performance of the proposed FuseCPath framework with other baseline methods, we utilize the publicly available datasets in the Cancer Genome Atlas (TCGA) [46] on three essential downstream tasks, which are biomarker prediction, gene expression prediction, and survival analysis. The statistics of the WSIs corresponding to diferent biomarker mutations in TCGA-BLCA, TCGA-LUAD, and TCGA-COAD datasets are summarized in Table 1. Additionally, we present the examples of WSI and corresponding patches utilized in our datasets in Figure 4.

![](images/cfb1c49c7f84aa63469820383b8e9df2af0064c70a3056f1a1da115bba1f4430.jpg)  
Fig. 4 Examples of WSIs in TCGA-COAD dataset for mutation and wild type. Several patches are selected for visualization.

Table 1 The statistics of the WSIs corresponding to diferent biomarker mutations in TCGA-BLCA, TCGA-LUAD, and TCGA-COAD datasets.

<table><tr><td rowspan="2">TCGA Biomarkers</td><td colspan="2">BLCA</td><td colspan="5">LUAD</td><td colspan="2">COAD</td></tr><tr><td>TP53</td><td>ATM</td><td>EGFR</td><td>FAT1</td><td>KRAS</td><td>LRP1B</td><td>TP53</td><td>BRAF</td><td>KRAS</td></tr><tr><td>Mutation</td><td>196</td><td>57</td><td>74</td><td>51</td><td>149</td><td>185</td><td>210</td><td>62</td><td>183</td></tr><tr><td>Wild-Type</td><td>210</td><td>349</td><td>483</td><td>506</td><td>408</td><td>371</td><td>347</td><td>366</td><td>245</td></tr><tr><td>Total</td><td>406</td><td>406</td><td>557</td><td>557</td><td>557</td><td>557</td><td>557</td><td>428</td><td>428</td></tr></table>

TCGA-LUAD. The lung adenocarcinoma cancer (LUAD) dataset contains 557 WSIs. 445 of them are selected as the training dataset, and 112 of them are selected as the validation dataset. The tasks of biomarker prediction for EGFR, FAT1, KRAS, LRP1B, and TP53 are utilized in our experiments. The corresponding survival times and censor state are provided for training.

TCGA-BLCA. The bladder cancer (BLCA) dataset contains 406 WSIs. 324 of them are selected as the training dataset, and 82 of them are selected as the validation dataset. The tasks of biomarker prediction for TP53 and ATM are utilized in our experiments. The corresponding survival times and censor state are provided for training.

TCGA-COAD. The colon adenocarcinoma cancer (COAD) dataset contains 428 WSIs. 342 of them are selected as the training dataset, and 86 of them are selected as the validation dataset.

The tasks of biomarker prediction for BRAF and KRAS are utilized in our experiments.

Metrics for biomarker prediction. The WSI-based biomarker prediction task can be modeled as a binary classification problem. Current studies typically evaluate the WSI-based classification methods using the Area Under the Receiver Operating Characteristic Curve (AUROC) metric. The AUROC is particularly suitable for the classification task. It provides a reliable assessment of classifier performance that accounts for both positive and negative sample classification across all decision thresholds, making it robust even with imbalanced data distributions.

Metrics for gene expression prediction. The WSI-based gene expression prediction can be modeled as a regression task. The prediction result is a vector containing each expression of the target gene. The prediction results $( { \bf y } _ { p r e d } )$ and ground truth (y) are regarded as the input of MSE, which is formulated as follows:

$$
\operatorname{MSE} \left(\mathbf {y} _ {\text { p   r   e   d }}, \mathbf {y}\right) = \frac {1}{N} \sqrt {\sum_ {i = 1} ^ {N = 1 0} \left\| \mathbf {y} _ {\text { p   r   e   d }} ^ {i} - \mathbf {y} ^ {i} \right\| ^ {2}},\tag{15}
$$

Metrics for survival analysis. To evaluate the performance of survival analysis, we select the metric called the Concordance Index (C-index) for our comparisons. C-index measures the concordance of the ranking for predicted risk with the ground truth survival times, which is formulated as follows:

Table 2 Comparisons of our proposed FuseCPath framework with diferent MIL-based methods to predict biomarkers on the TCGA-LUAD and TCGA-BLCA datasets. The bold results denote the highest scores, and the underlined results denote the second-highest scores.

<table><tr><td rowspan="2">AUROCMethods</td><td colspan="4">TCGA-LUAD</td><td colspan="2">TCGA-BLCA</td><td rowspan="2">Average</td></tr><tr><td>EGFR</td><td>FAT1</td><td>KRAS</td><td>LRP1B</td><td>TP53</td><td>ATM</td></tr><tr><td>MeanMIL</td><td> $80.8 \pm 0.8$ </td><td> $78.7 \pm 1.7$ </td><td> $78.6 \pm 0.5$ </td><td> $79.2 \pm 3.4$ </td><td> $77.4 \pm 0.8$ </td><td> $80.4 \pm 0.6$ </td><td> $79.2 \pm 1.3$ </td></tr><tr><td>MaxMIL</td><td> $80.1 \pm 1.2$ </td><td> $79.3 \pm 1.4$ </td><td> $80.1 \pm 1.0$ </td><td> $80.0 \pm 0.6$ </td><td> $78.0 \pm 1.1$ </td><td> $79.9 \pm 1.0$ </td><td> $79.6 \pm 1.1$ </td></tr><tr><td>AB-MIL [32]</td><td> $82.9 \pm 1.3$ </td><td> $81.0 \pm 2.7$ </td><td> $81.5 \pm 0.3$ </td><td> $80.3 \pm 4.9$ </td><td> $81.0 \pm 1.9$ </td><td> $82.1 \pm 1.2$ </td><td> $81.5 \pm 2.1$ </td></tr><tr><td>TransMIL [47]</td><td> $83.2 \pm 1.3$ </td><td> $81.1 \pm 0.7$ </td><td> $82.0 \pm 1.7$ </td><td> $81.8 \pm 0.9$ </td><td> $82.4 \pm 1.0$ </td><td> $83.7 \pm 1.1$ </td><td> $82.4 \pm 1.1$ </td></tr><tr><td>R2T-MIL [36]</td><td> $86.4 \pm 1.2$ </td><td> $83.2 \pm 0.3$ </td><td> $84.9 \pm 0.8$ </td><td> $84.2 \pm 1.1$ </td><td> $84.5 \pm 0.8$ </td><td> $85.9 \pm 0.8$ </td><td> $84.9 \pm 0.8$ </td></tr><tr><td>FuseCPath (Ours)</td><td> $89.5 \pm 0.7$ </td><td> $85.8 \pm 1.2$ </td><td> $86.8 \pm 0.1$ </td><td> $86.4 \pm 1.0$ </td><td> $86.0 \pm 1.1$ </td><td> $88.3 \pm 0.6$ </td><td> $87.1 \pm 0.8$ </td></tr></table>

$$
C _ {\text { index }} = \frac {1}{n} \sum_ {i \in \{i, \dots , n | \delta_ {i} = 1 \}} \sum_ {t _ {i} > t _ {j}} I [ f _ {i} > f _ {j} ].\tag{16}
$$

where n denotes the number of pairs for comparisons. I [·] denotes the indicator function. t denotes the observed survival time. f denotes the corresponding predicted risk. The value of the C-index ranges from 0 to 1. A higher C-index presents a better survival prognosis and vice versa. When the C-index value is 0.5, the prediction is inefective.

## 4.2 Biomarker predictions

Another comparison metric is the univariate Kaplan-Meier survival curve with log-rank p-values. In survival analysis, the disease state changes over time. The Kaplan-Meier survival curve intuitively demonstrates the survival differences of patients in diferent groups, and the log-rank method can be further used to test the statistical significance of the diferences.

Metrics for clustering. Due to the ground truth labels for the evaluation of clustering being unavailable, we select two widely adopted metrics for evaluating the clustering, which are the silhouette coeficient (SC) and the Calinski-Harabasz (CH) index. The value of SC ranges from -1 to 1, where values approaching 0 suggest cluster overlap, negative values indicate incorrect assignments of the clusters, and higher positive values reflect well-separated clusters. The CH index evaluates clustering quality by calculating the ratio of between-cluster variance to within-cluster variance, where variance is defined as the sum of squared Euclidean distances. Higher CH values indicate better clustering results, reflecting both strong separation between diferent clusters and high compactness within individual clusters.

Comparisons with MIL-based methods. In this experiment, we perform a comprehensive evaluation of our FuseCPath framework against previous MIL-based methods, including Mean-MIL, MaxMIL, AB-MIL, TransMIL, and vanilla R<sup>2</sup>T-MIL. For a fair comparison, each MIL-based method is trained using fused foundation model features from CONCH [9], Virchow [10], and Gigapath [14] through a direct concatenation strategy.

To quantitatively evaluate the performance of each method, we present the results assessed by AUROC in Table 2. The experimental results demonstrate that the proposed FuseCPath framework outperforms these baseline MIL-based methods. Performance improvements are observed across multiple biomarker prediction tasks on datasets TCGA-LUAD and TCGA-BLCA, with an average increase of 5% in AUROC compared to the baseline methods with the best performance. The superior results can be attributed to the teacher model’s high-level feature representations, which provide additional discriminative information to guide the student models’ feature fusion process. These results prove that our re-embedding and distillation-based FuseCPath framework enhances feature learning, particularly in scenarios with class imbalance and limited labeled training data.

Comparisons with individual FMs. In this experiment, we evaluate the classification performance of our proposed FuseCPath framework against state-of-the-art (SOTA) FMs across two biomarker prediction tasks BRAF and KRAS predictions in the TCGA-COAD dataset. To ensure a fair comparison, we have reproduced all baseline methods using their embeddings with the same classifier implementation.

The comprehensive evaluation results are presented in Table 3, which is measured by AUROC,

![](images/3f2849e5f79cb7ba233ae9afac91593c85ad937fa37e400f0fb687ea5969ddfa.jpg)  
Fig. 5 Visualized results of the gene expression predictions and the ground truth values observed by bulk RNA sequencing. If the values are closer to the ground truth, the prediction results are better.

Table 3 Comparisons of our proposed FuseCPath framework with diferent embeddings from the FMs to predict biomarkers on the TCGA-LUAD and TCGA-COAD datasets. The bold results denote the highest scores, and the underlined results denote the second-highest scores.

<table><tr><td rowspan="2">AUROCMethods</td><td colspan="2">TCGA-COAD</td><td rowspan="2">Average</td></tr><tr><td>BRAF</td><td>KRAS</td></tr><tr><td>CTransPath ([48])</td><td>58.8±5.5</td><td>52.5±9.1</td><td>55.7±7.3</td></tr><tr><td>Virchow ([49])</td><td>62.7±2.7</td><td>47.8±9.6</td><td>55.3±6.2</td></tr><tr><td>CONCH ([9])</td><td>59.4±3.0</td><td>55.3±6.1</td><td>57.4±4.6</td></tr><tr><td>H-Optimus ([50])</td><td>84.7±0.7</td><td>49.6±5.7</td><td>67.2±3.2</td></tr><tr><td>UNI ([13])</td><td>73.4±3.1</td><td>56.7±4.5</td><td>65.1±3.8</td></tr><tr><td>Gigapath ([14])</td><td>76.7±4.5</td><td>61.4±8.1</td><td>69.1±6.3</td></tr><tr><td>Virchow2 ([10])</td><td>83.0±2.6</td><td>60.9±1.8</td><td>71.9±2.2</td></tr><tr><td>Gigapath-SE ([14])</td><td>50.0±5.1</td><td>51.8±4.4</td><td>51.0±4.9</td></tr><tr><td>MADELEINE ([51])</td><td>58.4±1.9</td><td>53.6±3.3</td><td>56.0±2.6</td></tr><tr><td>CHIEF ([12])</td><td>67.1±5.1</td><td>56.9±8.7</td><td>62.0±6.9</td></tr><tr><td>PRISM ([11])</td><td>57.2±1.9</td><td>57.1±7.6</td><td>57.2±3.8</td></tr><tr><td>COBRA ([7])</td><td>86.2±2.8</td><td>58.1±6.9</td><td>72.3±4.9</td></tr><tr><td>FuseCPath (Ours)</td><td>91.8±3.0</td><td>78.1±5.4</td><td>84.9±4.2</td></tr></table>

reveal several key findings: First, FuseCPath consistently outperforms all individual FMs across the prediction tasks on TCGA-COAD. The observed average performance has improved by 17%. This improvement can be attributed to two fundamental advantages of our FuseCPath framework: First, the efective fusion of complementary features from heterogeneous FMs through our proposed re-embedding and distillation mechanism. Second, the patch-level and slide-level simultaneous fusion of multiple FMs adaptively emphasizes the most meaningful features for the prediction of each biomarker. These results imply that an efective fusion of diverse FMs can yield superior predictive capability compared to a single model, as the ensemble approach mitigates individual model limitations while preserving their respective strengths through feature fusion. The results also prove the importance of the fusion of pathology FMs, demonstrating that a carefully devised fusion framework can improve the model’s perfor mance by leveraging the rich but complementary information contained in diverse FMs.

## 4.3 Gene expression prediction

In this experiment, we evaluate the performance of gene expression prediction for our method. The proposed FuseCPath is capable of predicting the expression of many genes involved in pathways. We select 10 popularly investigated genes to assess the prediction errors of the proposed FuseCPath, which are TP53, EGFR, KRAS, BRAF, PIK3CA, IDH1, FGFR3, RB1, ATM, and ERBB2. Expressions are evaluated by logarithmically transformed transcripts per million (TPM) values t, which are formulated as follows:

$$
t = \log_ {2} (\mathrm{TPM} + 1).\tag{17}
$$

We select two methods for our comparisons, which are the proposed FuseCPath (wdist) and the method only containing patch-level features using R<sup>2</sup>T without the distillation module (w/o dist). Figure 5 presents radar charts comparing predicted and ground truth gene expression, visually illustrating the alignment between our model’s predictions and ground truths. We present the quantitative results averaged across all samples from the validation datasets of TCGA-LUAD, TCGA-BLCA, and TCGA-COAD in Table 4, respectively. To evaluate the prediction error, we provide the mean square error (MSE) comparisons between the prediction results of diferent methods and the ground truth in Table 5, demonstrating the accuracy of the prediction for individual genes.

Table 4 The quantitative results of gene expression prediction. The expressions are calculated by Equation 17. If the values are closer to the ground truth, the prediction results are better.

<table><tr><td>Genetypes / Datasets</td><td></td><td>TP53</td><td>EGFR</td><td>KRAS</td><td>BRAF</td><td>PIK3CA</td><td>IDH1</td><td>FGFR3</td><td>RB1</td><td>ATM</td><td>ERBB2</td></tr><tr><td rowspan="3">TCGA-LUAD</td><td>g.t.</td><td>1.961</td><td>1.954</td><td>1.919</td><td>0.651</td><td>0.706</td><td>2.875</td><td>1.224</td><td>1.705</td><td>0.869</td><td>2.661</td></tr><tr><td>w/o dist</td><td>2.310</td><td>2.205</td><td>2.206</td><td>0.711</td><td>0.668</td><td>3.342</td><td>1.520</td><td>1.844</td><td>0.997</td><td>2.961</td></tr><tr><td>wdist</td><td>2.318</td><td>2.158</td><td>2.188</td><td>0.703</td><td>0.662</td><td>3.307</td><td>1.518</td><td>1.807</td><td>0.981</td><td>2.951</td></tr><tr><td rowspan="3">TCGA-BLCA</td><td>g.t.</td><td>2.144</td><td>2.016</td><td>1.814</td><td>0.739</td><td>0.766</td><td>3.214</td><td>2.475</td><td>1.736</td><td>0.652</td><td>2.869</td></tr><tr><td>w/o dist</td><td>2.516</td><td>2.305</td><td>2.045</td><td>0.913</td><td>0.814</td><td>3.693</td><td>3.033</td><td>1.854</td><td>0.723</td><td>3.333</td></tr><tr><td>wdist</td><td>2.486</td><td>2.270</td><td>2.004</td><td>0.890</td><td>0.822</td><td>3.612</td><td>3.029</td><td>1.842</td><td>0.735</td><td>3.270</td></tr><tr><td rowspan="3">TCGA-COAD</td><td>g.t.</td><td>2.375</td><td>1.462</td><td>1.783</td><td>0.432</td><td>0.478</td><td>3.004</td><td>1.625</td><td>1.870</td><td>0.551</td><td>2.571</td></tr><tr><td>w/o dist</td><td>2.706</td><td>1.575</td><td>2.150</td><td>0.361</td><td>0.806</td><td>3.330</td><td>1.598</td><td>2.307</td><td>0.673</td><td>2.889</td></tr><tr><td>wdist</td><td>2.709</td><td>1.542</td><td>2.105</td><td>0.369</td><td>0.759</td><td>3.281</td><td>1.637</td><td>2.273</td><td>0.658</td><td>2.887</td></tr></table>

Table 5 Performance comparison on the gene expression prediction with diferent methods, which are evaluated with Mean Squared Error (MSE). The bold results denote the best scores. Lower values are closer to the ground truth.

<table><tr><td>Methods</td><td>TCGA-LUAD</td><td>TCGA-BLCA</td><td>TCGA-COAD</td></tr><tr><td>w/o dist</td><td>0.265</td><td>0.329</td><td>0.280</td></tr><tr><td>wdist</td><td>0.250</td><td>0.298</td><td>0.256</td></tr></table>

From these results, we can conclude that FuseCPath efectively predicts gene expression using the embeddings containing enough knowledge distilled from multiple FMs, without requiring additional specialized knowledge from genomics training data. Additionally, the average prediction error of the complete model of FuseC-Path remains below 30% for all genes in this experiment, indicating consistent performance across diferent genetic targets and types of cancers. The findings indicate that the distillation mechanism enables more eficient utilization of useful information contained in multiple FMs.

## 4.4 Survival Analysis

In this experiment, we evaluate and compare the performance of the proposed FuseCPath with the features from several slide-level FMs, which are CHIEF [12], Gigapath-SE [14], and PRISM [11]. We provide the results of Kaplan-Meier survival curves for each comparison method in Figure 6. The test cohorts are divided into high- and lowrisk groups using the median risk score predicted by our proposed FuseCPath framework. Comparative analysis demonstrates that FuseCPath achieves significantly improved risk stratification, producing a more distinct separation between the two risk groups with enhanced prognostic discrimination capability. This implies that the FuseC-Path consistently performs better than the other FMs in distinguishing high- and low-risk patients.

Table 6 Performance comparison on the survival analysis with diferent slide-level foundation model features, which are evaluated with C-index. The bold results denote the highest scores and the underlined results denote the second-highest scores.

<table><tr><td>Methods</td><td>TCGA-BLCA</td><td>TCGA-LUAD</td></tr><tr><td>CHIEF</td><td> $0.614 \pm 0.037$ </td><td> $0.644 \pm 0.052$ </td></tr><tr><td>Gigapath-SE</td><td> $0.649 \pm 0.039$ </td><td> $0.659 \pm 0.050$ </td></tr><tr><td>PRISM</td><td> $0.629 \pm 0.036$ </td><td> $0.634 \pm 0.046$ </td></tr><tr><td>FuseCPath (Ours)</td><td> $0.706 \pm 0.031$ </td><td> $0.708 \pm 0.049$ </td></tr></table>

To further evaluate the performance of FuseC-Path, we conduct quantitative experiments using the metric C-index, and the results are presented in Table 6. Compared with the stateof-the-art slide-level FMs, we can find that the performance of FuseCPath is the highest value in C-index among all comparison methods. The C-index is improved by 8.8% and 7.4% on the dataset TCGA-BLCA and TCGA-LUAD over the second-best method, respectively. The prediction performance of the proposed FuseCPath is better than that of individual FMs, which implies that the model will benefit from the knowledge from both patch-level and slide-level embeddings.

## 4.5 Analysis of clustering

Before the training process of the FuseCPath, one essential step is to select representative image patches from the input WSIs. In this work, to integrate the features from heterogeneous patchlevel FMs, we devise a multi-view clustering-based strategy to partition the patches into K = 50 clusters. Each embedding from the corresponding foundation model can be regarded as a view of the

![](images/536596b1be3168d193c65ce593261817791ee78212e794301c672a256e4ab5bc.jpg)  
Fig. 6 Kaplan-Meier survival curves of FuseCPath and representative slide-level FMs on TCGA-BLCA and TCGA-LUAD datasets.

WSI. In this section, we conduct an experimental analysis of clustering.

Visualization and interpretability. We present the visualized results of the clustering for each example WSI in Figure 7. To better demonstrate the visualization results, we present the zoom-in areas in original WSIs alongside their corresponding clustering results. The results clearly show that under the guidance of embeddings from the heterogeneous FMs, the clustering results exhibit clear alignment with cellular morphological distributions. The clustered regions are related to tissue structures, indicating that the multi-view clustering can capture both local and global morphological patterns. This implies that the patches selected from clusters are representative and reliable enough for the training of FuseCPath.

Analysis of the number of clusters. To determine the optimal number of clusters (K) for our multi-view clustering, we performed a systematic evaluation guided by both quantitative metrics and biological considerations. To ensure an adequate representation of histological patterns, we set a lower bound of 30 clusters in this experiment. As fewer clusters probably afect the diversity of selected patches. We validate this range by evaluating the quality of the cluster at 10-cluster intervals in Table 7. The selected metrics are the silhouette coeficient (SC) and Calinski-Harabasz (CH). From the results, we can conclude that $K \ = \ 5 0$ emerges as the optimal choice that satisfies both our computational metrics and the visual constraints of biological tissues. So we ultimately selected K = 50 clusters for the training of FuseCPath.

<table><tr><td># Clusters</td><td>SC</td><td>CH</td></tr><tr><td>K=30</td><td>0.09</td><td>832.5</td></tr><tr><td>K=40</td><td>0.10</td><td>946.1</td></tr><tr><td>K=50</td><td>0.15</td><td>1025.6</td></tr><tr><td>K=60</td><td>0.13</td><td>997.6</td></tr><tr><td>K=70</td><td>0.11</td><td>879.4</td></tr></table>

Table 8 Performance comparison of clustering methods, which are evaluated with SC and CH.

<table><tr><td>Clustering methods</td><td>SC</td><td>CH</td></tr><tr><td>Spectral Clustering</td><td>0.07</td><td>209.1</td></tr><tr><td>Agglomerative Clustering</td><td>0.07</td><td>893.3</td></tr><tr><td>Affinity Propagation</td><td>0.09</td><td>713.8</td></tr><tr><td>Multi-view Clustering</td><td>0.15</td><td>1025.6</td></tr></table>

![](images/a8b3753dfef5d77802be37ae08a79e83f9c0367e7a0648aa3c52af19c4cf04a0.jpg)  
Fig. 7 Visualized results of patch multi-view clustering (K = 50) based on patch embeddings derived from heterogeneous FMs.

Multi-view clustering vs. single-view clustering. In this part, we devise experiments to compare multi-view clustering with other singleview clustering. We select spectral clustering, agglomerative clustering, and afinity propagation for comparisons. These selected methods are clustered with embeddings from CONCH [9]. We present SC and CH quantitative results in Table 8. The results prove that multi-view clustering achieves a higher quality of clustering compared to the other single-view clustering methods. Multiview clustering integrates complementary features from heterogeneous patch-level FMs into a unified representation space, whereas single-view clustering, such as spectral clustering, only operates on a single feature space. Consequently, multiview clustering will capture higher-level relationships between diferent feature spaces, enabling nonlinear pattern discovery beyond single-view clustering limitations.

Table 9 Ablation study on the number of teacher models (Slide-level FMs) to predict biomarker TP53 on the TCGA-LUAD and TCGA-BLCA datasets. FuseCPath<sup>+</sup> indicates that the slide-level distillation module is eliminated from the complete framework. G denotes Gigapath-SE. P denotes PRISM. T denotes TITAN.

<table><tr><td>AUROCMethods</td><td>LUADTP53</td><td>BLCATP53</td><td>Average</td></tr><tr><td>FuseCPath $^{+}$  (0FM)</td><td>85.7</td><td>83.0</td><td>84.4</td></tr><tr><td>FuseCPath $^{+}$ +G (1FM)</td><td>86.5</td><td>83.6</td><td>85.1</td></tr><tr><td>FuseCPath $^{+}$ +G+P (2FMs)</td><td>87.2</td><td>85.2</td><td>86.2</td></tr><tr><td>FuseCPath $^{+}$ +G+P+T (3FMs)</td><td>89.5</td><td>86.0</td><td>87.8</td></tr></table>

Table 10 Ablation study on the number of patches for training to predict biomarker TP53 on the TCGA-LUAD and TCGA-BLCA datasets.

<table><tr><td>AUROC# Patches</td><td>LUADTP53</td><td>BLCATP53</td><td>Average</td></tr><tr><td> $N_{K}=300$ </td><td>86.7</td><td>83.2</td><td>85.0</td></tr><tr><td> $N_{K}=400$ </td><td>87.8</td><td>85.7</td><td>86.8</td></tr><tr><td> $N_{K}=500$ </td><td>89.5</td><td>86.0</td><td>87.8</td></tr><tr><td> $N_{K}=600$ </td><td>88.6</td><td>84.7</td><td>86.9</td></tr><tr><td> $N_{K}=700$ </td><td>88.1</td><td>85.5</td><td>86.8</td></tr></table>

## 4.6 Ablation studies

To evaluate the efectiveness of the main components of our proposed FuseCPath framework, we conduct ablation studies on the following aspects: the number of slide-level FMs and the number of selected patches. All experiments were conducted on the prediction task of biomarker TP53 for TCGA-LUAD and TCGA-BLCA datasets.

Ablation studies on the number of slidelevel FMs. In this experiment, we conduct an ablation study on the number of slide-level FMs for distillation. We present the quantitative results of AUROC in Table 9 and Figure 8. The best prediction performance (AUROC) is obtained when 3 slide-level FMs are utilized. The performance is decreasing when the number of slide-level FMs decreases. When FuseCPath is only trained with re-embedded patch-level features, the performance decreases by 3.6% on average. These results imply that more slide-level FMs utilized for distillation will provide more useful semantic information during training.

![](images/2d5d42dbc6d23fd6b605bebe3b783f3624ea880cdcd1dc92d71b484988be170f.jpg)  
Fig. 8 Ablation study on the number of teacher models (Slide-level FMs). The AUROC is increasing with the number of teacher models.

![](images/fe5691b6d4d11a71dadbd465061eec30ac6e564ac1f1fc903b424ff1c43913e3.jpg)  
Fig. 9 Ablation studies on the number of selected patches $( N _ { K } )$ for training. The best performance measured by AUROC is observed by $N _ { K } { = } 5 0 0$

Ablation studies on the number of selected patches. In this experiment, we conduct an ablation study on the number of selected patches $N _ { K }$ for patch-level features re-embedding during training. We present the quantitative results of AUROC in Table 10 and Figure 9. The best prediction performance (AUROC) is obtained when $N _ { K } { = } 5 0 0$ , which means that 10 patches are selected from 50 clusters in total. When $N _ { K } < 5 0 0$ the performance will increase as more patches are selected for training, this is because more patches will summarize the semantic information of WSIs more comprehensively. When $N _ { K } > 5 0 0$ the performance will slightly decrease because of the overfitting problem. Consequently, we select $N _ { K } = 5 0 0$ to implement the proposed FuseCPath framework.

## 5 Discussion and limitation

In this work, we have proposed a novel framework called FuseCPath for the fusion of multi-scale heterogeneous pathology FMs simultaneously. The proposed FuseCPath framework includes the following essential modules to efectively fuse the pathology FMs: representative patches selection based on multi-view patch features clustering, patch-level features re-embedding & aggregation, and slide-level collaborative distillation. These modules contribute to the performance improvement of biomarker prediction, gene expression, and survival analysis on datasets TCGA-LUAD, TCGA-BLCA, and TCGA-COAD. In conclusion, the FuseCPath framework will yield a new ensemble model with superior performance and benefit many meaningful downstream tasks. Additionally, clustering with multi-view features will provide insight into the visualization analysis of tissue morphography in WSI analysis.

Despite the demonstrated utility in this article, the FuseCPath poses potential limitations in its capacity to integrate more high-dimensional multi-omics data, such as the emerging spatial transcriptomic technologies. The current framework may not fully capture the underlying nonlinear relationships between diferent omics data. The rapid evolution of FMs presents a promising strategy for the integration of multi-omics data and WSIs [28]. In future research, we will extend the FuseCPath framework to the fusion of multi-omics representations and embeddings from multi-omics FMs to improve the precision of molecular-level WSI analysis.

## Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## CRediT authorship contribution statement

Zhidong Yang: Methodology, Investigation, Writing - original draft & editing. Xiuhui

Shi: Conceptualization, Resources, Data curation. Wei Ba: Conceptualization, Data curation. Zhigang Song: Conceptualization, Data curation. Haijing Luan: Methodology, Validation. Taiyuan Hu: Methodology, Validation. Senlin Lin: Conceptualization, Validation. Jiguang Wang: Resources, Writing - review & editing. Shaohua Kevin Zhou: Resources, Writing - review & editing. Rui Yan: Methodology, Resources, Writing - review & editing.

## Acknowledgments

This study was funded by the National Natural Science Foundation of China (62402473, 62271465), Beijing Natural Science Foundation (L252175), and the Suzhou Basic Research Program (SYG202338). J.W. lab is supported by RGC grant (R6003–22), ITC grant (ITCPD/17- 9), Padma Harilela Professorship, and Sinovac Fellowship Program.

## References

[1] Lu, M.Y., Chen, R.J., Kong, D., Lipkova, J., Singh, R., Williamson, D.F.K., Chen, T.Y., Mahmood, F.: Federated learning for computational pathology on gigapixel whole slide images. Medical Image Analysis 76, 102298 (2022) https://doi.org/10.1016/j.media.2021. 102298

[2] Huang, Y., Zhao, W., Chen, Y., Fu, Y., Yu, L.: Free lunch in pathology foundation model: Task-specific model adaptation with concept-guided feature enhancement. In: The Thirty-eighth Annual Conference on Neural Information Processing Systems (2024). https://doi.org/https://openreview. net/forum?id=dwYekpbmYG

[3] Yan, R., Lv, Z., Yang, Z., Lin, S., Zheng, C., Zhang, F.: Sparse and hierarchical transformer for survival analysis on whole slide images. IEEE Journal of Biomedical and Health Informatics 28(1), 7–18 (2024) https: //doi.org/10.1109/JBHI.2023.3307584

[4] Jaume, G., Vaidya, A., Chen, R., Williamson, D., Liang, P., Mahmood, F.: Modeling dense multimodal interactions between biological

pathways and histology for survival prediction. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024)

[5] Yan, R., Zhang, X., Jiang, Z., Wang, B., Bian, X., Ren, F., Zhou, S.K.: Pathwayaware multimodal transformer (pamt): Integrating pathological image and gene expression for interpretable cancer survival analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025)

[6] Yan, R., Shen, Y., Zhang, X., Xu, P., Wang, J., Li, J., Ren, F., Ye, D., Zhou, S.K.: Histopathological bladder cancer gene mutation prediction with hierarchical deep multiple-instance learning. Medical Image Analysis 87, 102824 (2023) https://doi.org 10.1016/j.media.2023.102824

[7] Lenz, T., Neidlinger, P., Ligero, M., Wolflein, G., Van Treeck, M., Kather, J.N.: Unsupervised foundation modelagnostic slide-level representation learning. In: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 30807–30817 (2025). https: //doi.org/10.1109/CVPR52734.2025.02869

[8] Luan, H., Hu, T., Hu, J., Liu, W., Yang, K., Pei, Y., Li, R., He, J., Gao, Y., Sun, D., Duan, X., Yan, R., Zhou, S.K., Niu, B.: Breast cancer homologous recombination deficiency prediction from pathological images with a suficient and representative transformer. npj Precision Oncology 9, 160 (2025)

[9] Lu, M.Y., Chen, B., Williamson, D.F., Chen, R.J., Liang, I., Ding, T., Jaume, G., Odintsov, I., Le, L.P., Gerber, G., Parwani, A.V., Zhang, A., Mahmood, F.: A visual-language foundation model for computational pathology. Nature Medicine 30, 863–874 (2024)

[10] Zimmermann, E., Vorontsov, E., Viret, J., Casson, A., Zelechowski, M., Shaikovski, G., Tenenholtz, N., Hall, J., Fuchs, T., Fusi, N., Liu, S., Severson, K.: Virchow2: Scaling selfsupervised mixed magnification models in pathology. arXiv preprint arXiv:2408.00738

(2024)

[11] Shaikovski, G., Casson, A., Severson, K., Zimmermann, E., Wang, Y.K., Kunz, J.D., Retamero, J.A., Oakley, G., Klimstra, D., Kanan, C., Hanna, M., Zelechowski, M., Viret, J., Tenenholtz, N., Hall, J., Fusi, N., Yousfi, R., Hamilton, P., Moye, W.A., Vorontsov, E., Liu, S., Fuchs, T.J.: PRISM: A Multi-Modal Generative Foundation Model for Slide-Level Histopathology (2024). https: //arxiv.org/abs/2405.10254

[12] Wang, X., Zhao, J., Marostica, E., Yuan, W., Jin, J., Zhang, J., Li, R., Tang, H., Wang, K., Li, Y., Wang, F., Peng, Y., Zhu, J., Zhang, J., Jackson, C.R., Zhang, J., Dillon, D., Lin, N.U., Sholl, L., Denize, T., Meredith, D., Ligon, K.L., Signoretti, S., Ogino, S., Golden, J.A., Nasrallah, M.P., Han, X., Yang, S., Yu, K.-H.: A pathology foundation model for cancer diagnosis and prognosis prediction. Nature 634, 970–978 (2024)

[13] Chen, R.J., Ding, T., Lu, M.Y., Williamson, D.F., Jaume, G., Chen, B., Zhang, A., Shao, D., Song, A.H., Shaban, M., Williams, M., Oldenburg, L., Weishaupt, L.L., Wang, J.J., Vaidya, A., Le, L.P., Gerber, G., Sahai, S., Williams, W., Mahmood, F.: Towards a general-purpose foundation model for computational pathology. Nature Medicine (2024)

[14] Xu, H., Usuyama, N., Bagga, J., Zhang, S., Rao, R., Naumann, T., Wong, C., Gero, Z., Gonz´alez, J., Gu, Y., Xu, Y., Wei, M., Wang, W., Ma, S., Wei, F., Yang, J., Li, C., Gao, J., Rosemon, J., Bower, T., Lee, S., Weerasinghe, R., Wright, B.J., Robicsek, A., Piening, B., Bifulco, C., Wang, S., Poon, H.: A whole-slide foundation model for digital pathology from real-world data. Nature (2024)

[15] Ding, T., Wagner, S.J., Song, A.H., Chen, R.J., Lu, M.Y., Zhang, A., Vaidya, A.J., Jaume, G., Shaban, M., Kim, A., Williamson, D.F.K., Chen, B., Almagro-Perez, C., Doucet, P., Sahai, S., Chen, C., Komura, D., Kawabe, A., Ishikawa, S., Gerber, G., Peng, T., Le, L.P., Mahmood, F.: Multimodal whole slide foundation model for

pathology. Nature Medicine (2025) https: //doi.org/10.1038/s41591-025-03982-3

[16] Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., HAZIZA, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski, P.: DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research (2024)

[17] He, K., Chen, X., Xie, S., Li, Y., Doll´ar, P., Girshick, R.: Masked autoencoders are scalable vision learners. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 16000– 16009 (2022)

[18] Neidlinger, P., El Nahhas, O.S.M., Muti, H.S., Lenz, T., Hofmeister, M., Brenner, H., Treeck, M., Langer, R., Dislich, B., Behrens, H.M., R¨ocken, C., Foersch, S., Truhn, D., Marra, A., Saldanha, O.L., Kather, J.N.: Benchmarking foundation models as feature extractors for weakly supervised computational pathology. Nature Bimedical Engineering (2025)

[19] Grill, J.-B., Strub, F., Altch´e, F., Tallec, C., Richemond, P.H., Buchatskaya, E., Doersch, C., Pires, B.A., Guo, Z.D., Azar, M.G., Piot, B., Kavukcuoglu, K., Munos, R., Valko, M.: Bootstrap your own latent a new approach to self-supervised learning. In: Proceedings of the 34th International Conference on Neural Information Processing Systems. NIPS ’20 (2020)

[20] Chen, T., Kornblith, S., Norouzi, M., Hinton, G.: A simple framework for contrastive learning of visual representations. In: Proceedings of the 37th International Conference on Machine Learning, vol. 119, pp. 1597–1607 (2020)

[21] Azizi, S., Culp, L., Freyberg, J., Mustafa, B., Baur, S., Kornblith, S., Chen, T., Tomasev, N., Mitrovi´c, J., Strachan, P., Mahdavi, S.S., Wulczyn, E., Babenko, B., Walker,

M., Loh, A., Chen, P.-H.C., Liu, Y., Bavishi, P., McKinney, S.M., Winkens, J., Roy, A.G., Beaver, Z., Ryan, F., Krogue, J., Etemadi, M., Telang, U., Liu, Y., Peng, L., Corrado, G.S., Webster, D.R., Fleet, D., Hinton, G., Houlsby, N., Karthikesalingam, A., Norouzi, M., Natarajan, V.: Robust and data-eficient generalization of self-supervised machine learning for diagnostic imaging. Nature Biomedical Engineering 7, 756–779 (2023) https://doi.org/10.1038/ s41551-023-01049-7

[22] Yang, Z., Wei, T., Liang, Y., Yuan, X., Gao, R., Xia, Y., Zhou, J., Zhang, Y., Yu, Z.: A foundation model for generalizable cancer diagnosis and survival prediction from histopathological images. Nature Communications 16, 2366 (2025) https://doi.org/10. 1038/s41467-025-57587-y

[23] Nechaev, D., Pchelnikov, A., Ivanova, E.: Hibou: A Family of Foundational Vision Transformers for Pathology (2024)

[24] Chen, R.J., Chen, C., Li, Y., Chen, T.Y., Trister, A.D., Krishnan, R.G., Mahmood, F.: Scaling vision transformers to gigapixel images via hierarchical self-supervised learning. In: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 16123–16134 (2022). https:// doi.org/10.1109/CVPR52688.2022.01567

[25] Lazard, T., Lerousseau, M., Decenci\`ere, E., Walter, T.: Giga-ssl: Self-supervised learning for gigapixel images. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pp. 4305–4314 (2023)

[26] Vaidya, A., Zhang, A., Jaume, G., Song, A.H., Ding, T., Wagner, S.J., Lu, M.Y., Doucet, P., Robertson, H., Almagro-Perez, C., Chen, R.J., ElHarouni, D., Ayoub, G., Bossi, C., Ligon, K.L., Gerber, G., Le, L.P., Mahmood, F.: Molecular-driven Foundation Model for Oncologic Pathology (2025). https: //arxiv.org/abs/2501.16652

[27] Jaume, G., Vaidya, A., Zhang, A., H. Song, A., J. Chen, R., Sahai, S., Mo, D., Madrigal,

E., Phi Le, L., Mahmood, F.: Multistain pretraining for slide representation learning in pathology. In: ECCV 2024, pp. 19–37 (2025)

[28] Chen, W., Zhang, P., Tran, T.N., Xiao, Y., Li, S., Shah, V.V., Cheng, H., Brannan, K.W., Youker, K., Lai, L., Fang, L., Yang, Y., Le, N.-T., Abe, J.-i., Chen, S.-H., Ma, Q., Chen, K., Song, Q., Cooke, J.P., Wang, G.: A visual–omics foundation model to bridge histopathology with spatial transcriptomics. Nature Methods 22, 1568–1582 (2025)

[29] Wang, S., Zhu, Y., Yu, L., Chen, H., Lin, H., Wan, X., Fan, X., Heng, P.-A.: Rmdl: Recalibrated multi-instance deep learning for whole slide gastric image classification. Medical image analysis 58, 101549 (2019)

[30] Lu, M.Y., Williamson, D.F., Chen, T.Y., Chen, R.J., Barbieri, M., Mahmood, F.: Data-eficient and weakly supervised computational pathology on whole-slide images. Nature Biomedical Engineering 5(6), 555– 570 (2021)

[31] Li, B., Li, Y., Eliceiri, K.W.: Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 14318–14328 (2021)

[32] Ilse, M., Tomczak, J., Welling, M.: Attentionbased deep multiple instance learning. In: International Conference on Machine Learning, pp. 2127–2136 (2018). PMLR

[33] Yao, J., Zhu, X., Jonnagaddala, J., Hawkins, N., Huang, J.: Whole slide images based cancer survival prediction using attention guided deep multiple instance learning networks. Medical Image Analysis 65, 101789 (2020)

[34] Schirris, Y., Gavves, E., Nederlof, I., Horlings, H.M., Teuwen, J.: Deepsmile: Contrastive self-supervised pre-training benefits msi and hrd classification directly from h&e whole-slide images in colorectal and breast cancer. Medical Image Analysis 79, 102464 (2022)

[35] Yang, S., Wang, Y., Chen, H.: Mambamil: Enhancing long sequence modeling with sequence reordering in computational pathology. In: Proceedings of Medical Image Computing and Computer Assisted Intervention (MICCAI 2024), pp. 296–306 (2024)

[36] Tang, W., Zhou, F., Huang, S., Zhu, X., Zhang, Y., Liu, B.: Feature re-embedding: Towards foundation model-level performance in computational pathology. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11343– 11352 (2024)

[37] Li, R., Yao, J., Zhu, X., Li, Y., Huang, J.: Graph cnn for survival analysis on whole slide pathological images. In: International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 174–182 (2018). Springer

[38] Raju, A., Yao, J., Haq, M.M., Jonnagaddala, J., Huang, J.: Graph attention multi-instance learning for accurate colorectal cancer staging. In: International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 529–539 (2020). Springer

[39] Hou, L., Samaras, D., Kurc, T.M., Gao, Y., Davis, J.E., Saltz, J.H.: Patch-based convolutional neural network for whole slide tissue image classification. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2424–2433 (2016)

[40] Adnan, M., Kalra, S., Tizhoosh, H.R.: Representation learning of histopathology images using graph neural networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 988–989 (2020)

[41] Wang, X., Chen, H., Gan, C., Lin, H., Dou, Q., Tsougenis, E., Huang, Q., Cai, M., Heng, P.-A.: Weakly supervised deep learning for whole slide lung cancer image analysis. IEEE Transactions on Cybernetics 50(9), 3950– 3962 (2019)

[42] Kalra, S., Tizhoosh, H.R., Choi, C., Shah, S., Diamandis, P., Campbell, C.J., Pantanowitz,

L.: Yottixel–an image search engine for large archives of histopathology whole slide images. Medical Image Analysis 65, 101757 (2020)

[43] Zhu, X., Yao, J., Zhu, F., Huang, J.: Wsisa: Making survival prediction from whole slide histopathological images. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 7234–7242 (2017)

[44] Xu, H., Clemenceau, J.R., Park, S., Choi, J., Lee, S.H., Hwang, T.H.: Spatial heterogeneity and organization of tumor mutation burden with immune infiltrates within tumors based on whole slide images correlated with patient survival in bladder cancer. Journal of Pathology Informatics 13, 100105 (2022)

[45] Zhang, A., Jaume, G., Vaidya, A., Ding, T., Mahmood, F.: Accelerating Data Processing and Benchmarking of AI Models for Pathology (2025). https://arxiv.org/abs/ 2502.06750

[46] Kandoth, C., McLellan, M.D., Vandin, F., Ye, K., Niu, B., Lu, C., Xie, M., Zhang, Q., McMichael, J.F., Wyczalkowski, M.A., Leiserson, M.D.M., Miller, C.A., Welch, J.S., Walter, M.J., Wendl, M.C., Ley, T.J., Wilson, R.K., Raphael, B.J., Ding, L.: Mutational landscape and significance across 12 major cancer types. Nature 502(7471), 333–339 (2013)

[47] Shao, Z., Bian, H., Chen, Y., Wang, Y., Zhang, J., Ji, X., Zhang, Y.: Transmil: transformer based correlated multiple instance learning for whole slide image classification. In: Proceedings of the 35th International Conference on Neural Information Processing Systems. NIPS ’21 (2021)

[48] Wang, X., Yang, S., Zhang, J., Wang, M., Zhang, J., Yang, W., Huang, J., Han, X.: Transformer-based unsupervised contrastive learning for histopathological image classification. Medical Image Analysis 81, 102559 (2022) https://doi.org/10.1016/j.media.2022. 102559

[49] Vorontsov, E., Bozkurt, A., Casson, A.,

Shaikovski, G., Zelechowski, M., Severson, K., Zimmermann, E., Hall, J., Tenenholtz, N., Fusi, N., Yang, E., Mathieu, P., Eck, A., Lee, D., Viret, J., Robert, E., Wang, Y.K., Kunz, J.D., Lee, M.C.H., Bernhard, J.H., Godrich, R.A., Oakley, G., Millar, E., Hanna, M., Wen, H., Retamero, J.A., Moye, W.A., Yousfi, R., Kanan, C., Klimstra, D.S., Rothrock, B., Liu, S., Fuchs, T.J.: A foundation model for clinical-grade computational pathology and rare cancers detection. Nature Medicine (2024) https://doi.org/10. 1038/s41591-024-03141-0

[50] Saillard, C., Jenatton, R., Llinares-L´opez, F., Mariet, Z., Cahan´e, D., Durand, E., Vert, J.-P.: H-optimus-0. https://github.com/bioptimus/releases/ tree/main/models/h-optimus/v0

[51] Jaume, G., Vaidya, A.J., Zhang, A., Song, A.H., Chen, R.J., Sahai, S., Mo, D., Madrigal, E., Le, L.P., Faisal, M.: Multistain pretraining for slide representation learning in pathology. In: European Conference on Computer Vision (2024). Springer