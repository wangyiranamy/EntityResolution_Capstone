Advanced Guide
==============

.. error::
   Probably due to a bug with sphinx 2.0, the reference to the paper is not
   properly rendered in html in the `References`_ section. However, the
   reference content is correct and this effect is purely aesthetic.

This guide explains the algorithm used in this package and each parameter in
the `~entity_resolver.main.EntityResolver` class in great details. This
algorithm is the same as described in [1]_.

Algorithm
---------

This section describes the core algorithm used in this package.

Overview
>>>>>>>>

The algorithm consists of three parts: blocking, relational bootstrapping, and
clustering.

Blocking is aimed to quickly filter out reference pairs that are potentially
the same entities. It passes through the reference list once. For each
reference, it compares this reference with each of the current buckets'
reprensentative references and if their distances are less than a pre-defined
threshold, assign this reference to that bucket. If no assignment is made, then
create a new bucket with this reference as reprensentative. Note that a
reference can be assigned to multiple buckets and the comparison function
should take as little time as possible. Finally, any pair of references in the
same bucket are considered potentially the same. In other words, references
**not** ending up in the same bucket will **never** be in the same cluster.

Relational bootstrapping tries to initialize some reference clusters so that
computation of relational similarity is more accurate. Contrary to blocking, it
amis to put references that are **certainly** the same entities into one
cluster. It passes through pairs of potentially the same references (blocking
result), and merge the two references to the same cluster if they are
**highly** similar and the number of pairs of references connected with them
through hyper-edges which are also highly similar is **greater than or equal
to** a pre-defined threshold. Usually two references are considered highly
similar if their attributes (all or most of) are exactly the same. A union-find
data structure is used for this task to achieve maximum efficiency.

Clustering is the last step to determine which references are indeed the same
entities. The algorithm is a modified version of the agglomerative clustering.
It uses a priority queue to store the cluster pairs that are potentially the
same. These cluster pairs are initialized with results of blocking where each
pair of references are replaced with the clusters they belong to after
relational bootstrapping if the clusters are distinct. The similarity of each
pair of clusters are used as their priority value in the priority queue. The
similarity of two clusters is a linear combination of their attribute
similarity and relational similarity.

   :attribute similarity: Max, min or average of attribute similarities of
    each pair of references in the two clusters. The attribute similarity of
    two references is a weighted average of the attribute similarities of
    each of their attribute values. Specific attribute similarities on
    attribute values implemented in this package are explained in `Attribute
    Similarity Functions`_.
   :relational similarity: Based on the neighboring clusters of the cluster
    pairs. A cluster is called a neighboring cluster of another it contains a
    reference that is connected with another reference in the other cluster
    through a hyper-edge. Specific relational similarities implemented in this
    package are explained in `Relational Similarity Functions`_. Uniqueness of
    a cluster may also be used in computing the relational similarity, which
    are included in `Uniqueness`_.

   .. note::
      Our implementation always includes the cluster itself as one of its
      neighboring clusters. A cluster may appear in the neighboring clusters
      multiple times by this definition and therefore a multiset can be used to
      account for such multiplicity.

At each iteration of the algorithm, the pair of clusters with the highest
similarity is popped from the priority queue. If their similarity is less than
a pre-defined threshold, the algorithm stops. Otherwise, these two clusters are
merged into a new cluster, and any pair of clusters that include either one of
the merged clusters are removed from the priority queue. If a cluster is
considered potentially the same with either one of the merged clusters before
merging, it is also considered potentially the same as the new cluster, and
they are inserted into the priority queue with similarity computed as well. For
any neighboring cluster of the new cluster, recompute and update its similarity
with each of the clusters that are potentially the same as itself.

.. attention::
   If a popped pair of clusters contain two references that are connected with
   a hyper-edge, these two clusters are not merged and the algorithm continues.
   This is called a "negative constraint". This is a reasonable constraint in
   most cases since usually co-occurances of references are distinct entities.

Finally each cluster of references are considered the same entities.

Attribute Similarity Functions
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Below is a list of attribute similarity functions implemented in this package.
Strings inside brackets are the names used for corresponding attribute
similarity functions when specified in ``attr_strategy``.

.. tip::
    Users may implement more custom attribute similarity functions and passed
    to the ``attr_strategy`` parameter. Libraries `textdistance <https://
    github.com/life4/textdistance>`_ and `py_stringmatching <http://
    anhaidgroup.github.io/py_stringmatching/v0.4.1/index.html>`_ provide more
    string similarity functions.

:Jaro (jaro): Compare two strings as whole. Refer to `Wikipedia <https://en.m.
 wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance#Jaro_Similarity>`_ for
 theoretical details.

:Jaro-Winkler (jaro_winkler): Compare two strings as whole. Refer to `Wikipedia
 <https://en.m.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance#
 Jaro_Similarity>`_ for theoretical details.

:Soft-Tfidf (stfidf): Compare two strings after tokenization. Refer to `this
 paper <https://www.aclweb.org/anthology/C08-1075.pdf>`_ for theoretical
 details. Note that due to the assymetrical nature of the soft-tfidf algorithm,
 we compute the similarity by taking the maximum score of both permutations.

Relational Similarity Functions
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Below is a list of relational similarity functions implemented in this package.
Strings inside brackets are the names used for corresponding relational
similarity functions when specified in ``rel_strategy``.

:Jaccard Coefficient (jaccard_coef): It is equal to the number of clusters in
 the intersection of the two sets of neighboring clusters divided by the number
 of clusters in the union of the two sets:

 .. math::

    &\text{Jaccard Coefficient} = \frac{|S_1\cap S_2|}{|S_1\cup S_2|}\\
    &S_1 = \text{set of neighboring clusters of cluster1}\\
    &S_2 = \text{set of neighboring clusters of cluster2}\\
    &|S| = \text{number of elements in the set } S

 The neighboring cluster set does not account for multiplicities.

:Jaccard Coefficient With Multiplicity (jaccard_coef_fr): It is the same as
 *Jaccard Coefficient* except that the set of neighboring clusters is replaced
 with the multiset. Union of two multisets is a multiset of which the
 multiplicity of each element is the maximum of that in the two multiset,
 while intersection of two multisets is taking the minimum.

:Adar Similarity (adar_neighbor): Instead of counting the number of elements in
 union and intersection, one may use the uniqueness of clusters:

 .. math::

    &\text{Adar Similarity} = \frac{\sum_{c\in S_1\cap S_2}u(c)}{\sum_{c\in S_1\cup S_2}u(c)}\\
    &u(c) = \text{uniqueness of cluster} c

 The neighboring cluster set does not account for multiplicities. One can see
 that when the uniqueness of each cluster is 1, the above reduces to *Jaccard
 Coefficient*. The uniqueness is computed using ambiguity score using a
 neighbor-based approach, as explained in `Uniqueness`_.

:Adar Similarity With Multiplicity (adar_neighbor_fr): Use multiset instead of
 set as in *Adar Similarity* as well as when computing the uniqueness using the
 neighbor-based approach.

:Adar Similarity With Ambiguity (adar_attr): Same computation as *Adar
 Similarity*. The only difference is that it use an attribute-based approach to
 compute the uniqueness, as explained in `Uniqueness`_.

:Adar Similarity With Ambiguity & Multiplicity (adar_attr_fr): Use multiset
 instead of set in *Adar Similarity With Ambiguity*. Use the same approach to
 compute uniqueness.

Uniqueness
>>>>>>>>>>

:neighbor-based approach: Use the inverse of the log count of elements in the
 set (or multiset) of neighboring clusters:

 .. math::

    &\text{uniqueness of cluster } c = \frac{1}{1+\log(S)}\\
    &S = \text{set (or multiset) of neighboring clusters of } c

 We add 1 to the denominator to avoid zero division.
:attribute-based approach: Firstly one needs to compute two attributes
 :math:`A_1` and :math:`A_2` for each reference based on its existing
 attribute. Either of them can be the same as one of its existing attributes or
 a newly created one. For example, in the case of author names, :math:`A_1` may
 be the last name, and :math:`A_2` may be the first name initial. Then the
 ambiguity score of a paticular reference is equal to the number of distinct
 :math:`A_2` values in all references which have the same :math:`A_1` values as
 the one being evaluated divided by the total number of references:

 .. math::

    &\text{ambiguity of } r = \frac{|\sigma(\pi_{r'.A_2}(\delta_{r'.A_1=r.A_1}(R)))|}{|R|}\\
    &R = \text{the set of all references}\\
    &|S| = \text{number of elements in the set } S\\
    &\sigma = \text{take distinct values}\\
    &\pi = \text{project to an attribute}\\
    &\delta = \text{select elements from a set with certain constraints}

 Finally, the uniqueness of a cluster is computed by taking the inverse of the
 average ambiguity scores of all references contained in the cluster:

 .. math::

    &\text{uniqueness of } c = \frac{1}{\text{Avg}_{r\in c}(\text{Amb}(r))}\\
    &\text{Avg} = \text{calculate average score}\\
    &\text{Amb}(r) = \text{ambiguity score of } r

Evaluation Metrics
>>>>>>>>>>>>>>>>>>

Below is a list of all evaluation metrics implemented in this package. Strings
inside brackets are the names used for corresponding evaluation metric
functions when specified in ``evaluator_strategy``.

:precision, recall, & f1 (precision_recall): The formulae used for computing
 precision, recall, and f1 scores in the context of clustering are listed as
 follows:

 .. math::

    &precision = \frac{TP}{TP + FP}\\
    &recall = \frac{TP}{TP + FN}\\
    &f1 = \frac{2\times precision\times recall}{precision + recall}\\
    &TP = \frac{\text{number of pairs in the same cluster in both label and prediction}}{\text{total number of pairs}}\\
    &FP = \frac{\text{number of pairs in the same cluster in prediction but not label}}{\text{total number of pairs}}\\
    &FN = \frac{\text{number of pairs in the same cluster in label but not prediction}}{\text{total number of pairs}}

:ami: Refer to `scikit-learn documentation <https://scikit-learn.org/stable/
 modules/clustering.html#mutual-information-based-scores>`__ for theoretical
 details.

:v_measure: Refer to `scikit-learn documentation <https://scikit-learn.org/
 stable/modules/clustering.html#homogeneity-completeness-and-v-measure>`__ for
 theoretical details.

Parameters
----------

This section describes each parameter passed to construct an
`~entity_resolver.main.EntityResolver` object based on the comprehensive
description of the algorithm above. It is essential to understand the
`Algorithm`_ section before referring to this section for parameter details.

:alpha: The weight of relational similarity in computing the similarity of two
 references. In other words, :math:`\text{similarity} = (1-\alpha)\times\text{
 attribute similarity} + \alpha\times\text{relational similarity}`. Default is
 0.

:attr_strategy: It is a dictionary of attribute names mapping to strings or
 callables that describe how to compute the attribute similarities of two
 values beloinging to the corresponding attributes. It should take two
 attribute values. Valid strings are listed in `Attribute Similarity
 Functions`_. Users may also use custom functions that take two attribute
 values and return their similarity score. Such functions should accept two
 dictionaries of attribute names mapping to attribute values that represent a
 reference and return their similarity score. The attribute values are
 **always** preprocessed according to the attribute types. Note that the
 similarity scores on all attributes should be of the same scale since the
 final similarity score of two references is a weighted average of these. By
 default, the strategy is inferred from the attribute type.

 * ``'text'`` type is default to soft-tfidf.
 * ``'person_entity'`` type is default to Jaro-Winkler.
 * Other types must be accompanied by a custom callable in ``attr_strategy``.

:attr_types: Explained in :doc:`tutorial`.

:average_method: Refer to the ``**kwargs`` parameter in
 :doc:`complete_doc/entity_resolver.main`

:blocking_strategy: Explained in :doc:`tutorial`.

:blocking_threshold: Threshold used in blocking. Only pairs of references of
 distances computed by ``blocking_strategy`` **strictly less than** this value
 are considered potentially the same. Default is 3.

:bootstrap_strategy: Describes how the relational bootstrapping should be done
 by specifying when a pair of references should be considered as a bootstrap
 candidate (highly similar). Refer to its documentation in
 :doc:`complete_doc/entity_resolver.main` for technical details. Default is
 ``None`` and this implies two references should have exactly the same values
 for all attributes to be considered highly similar.

:edge_match_threshold: When considering if two references should be merged
 during relational bootstrapping, the number of pairs of references connected
 with these two through hyper-edges which are also highly similar must be
 **greater than or equal to** this value besides these two references
 themselves are highly similar. Default is 1.

:evaluator_strategy: Specify how to evaluate the performance of the entity
 resolution provided with groud truth data. Valid string values are listed in
 `Evaluation Metrics`_. It can also be a custom callable that follows the
 signatures of class methods in `~entity_resolver.core.utils.ClusteringMetrics`
 (two `~collections.OrderedDict` and ``**kwargs**`` as inputs and any
 performance indicator as output). Default is ``'precision_recall'``.

:first_attr: Describes how to compute the first attribute :math:`A_1` when
 calculating attribute-based uniqueness. It must be a valid callable if
 ``'adar_attr'`` or ``'adar_attr_fr'`` is used for ``rel_strategy``. A valid
 callable should accpet a dictionary mapping attribute names to attribute
 values that represent one reference and returns the generated attribute
 :math:`A_1`. Default is ``None``.

:first_attr_raw: Affects input of ``first_attr`` in the same way as how
 ``raw_blocking`` affects the input of ``blocking_strategy`` described in
 :doc:`tutorial`. Set ``first_attr`` to accept dictionaries consisting of raw
 attribute values. Default is ``False``.

:jw_prefix_weight: Refer to the ``**kwargs`` parameter in
 :doc:`complete_doc/entity_resolver.main`

:linkage: Describes how to compute the attribute similarity of two clusters
 based on the attribute similarities of references. Valid values are ``'max'``,
 ``'min'``, and ``'average'``. They stands for using max, min or average of
 attribute similarities of each pair of references in the two clusters
 respectively. Default is ``'max'``.

:plot_prc: Explained in :doc:`tutorial`.

:raw_blocking: Explained in :doc:`tutorial`.

:raw_bootstrap: Affects input of ``bootstrap_strategy`` in the same way as how
 ``raw_blocking`` affects the input of ``blocking_strategy`` described in
 :doc:`tutorial`. Set ``bootstrap_strategy`` to accept dictionaries consisting
 of raw attribute values. Default is ``False``.

:rel_strategy: A string that describes how to compute the relational similarity
 of two clusters. Valid strings are listed in `Relational Similarity
 Functions`_. Note that custom functions cannot be used here because we think
 it is difficult to define a uniform function signature for computation of
 relational similarity. Default is ``'jaccard_coef'``.

:second_attr: Similar to ``first_attr``, but it describes how to compute the
 second attribute :math:`A_2`. Default is ``None``.

:second_attr_raw: Similar to ``second_attr``, but it affects input of
 ``second_attr``. Default is ``None``.

:second_sim: Refer to the ``**kwargs`` parameter in
 :doc:`complete_doc/entity_resolver.main`

:seed: Explained in :doc:`tutorial`.

:similarity_threshold: It specifies the stopping criterion of the clustering
 algorithm. When the cluster pair popped from the priority queue is of
 similarity **strictly less than** the threshold, clustering stops.

:stfidf_threshold: Refer to the ``**kwargs`` parameter in
 :doc:`complete_doc/entity_resolver.main`

:verbose: Explained in :doc:`tutorial`.

:weights: The weights assigned to each attribute in computing the attribute
 similarity of two references. Default is ``None``, which automatically assigns
 equal weights to each attribute.


References
----------

.. [1] Indrajit Bhattacharya and Lise Getoor. 2007. Collective entity
   resolution in relational data. ACM Trans. Knowl. Discov. Data 1, 1, Article
   5 (March 2007). DOI=http://dx.doi.org/10.1145/1217299.1217304
