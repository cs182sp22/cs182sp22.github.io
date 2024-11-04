---
layout: home
title: Data C182 Fall 2024
nav_exclude: true
seo:
  type: Course
  name: Data C182 Fall 2024
---

# {{ site.tagline }}
{: .mb-2 }
{{ site.description }}
{: .fs-6 .fw-300 }

<!--{% if site.announcements %}-->
<!--{{ site.announcements.last }}-->
<!--[Announcements](announcements.md){: .btn .btn-outline .fs-3 }-->
<!--{% endif %}-->

## Course Description

From the [UC Berkeley course catalog](https://classes.berkeley.edu/content/2024-fall-data-c182-001-lec-001):  

Deep Networks have revolutionized computer vision, language technology, robotics and control.
They have growing impact in many other areas of science and engineering.
They do not, however, follow a closed or compact set of theoretical principles.
In Yann Lecun's words they require "an interplay between intuitive insights, theoretical modeling, practical implementations, empirical studies, and scientific analyses."
This course attempts to cover that ground.

**Important:** In this course, we will use [Edstem](https://edstem.org/us/courses/64085/discussion/) to post announcements and important information.
**It is the student's responsibility to actively monitor the Ed for any important announcements.**

**Useful course links:**
- bCourse: [link](https://bcourses.berkeley.edu/courses/1538180)
- Ed: [link](https://edstem.org/us/courses/64085/discussion/)
- Gradescope: [link](https://www.gradescope.com/courses/837491)

## Textbooks (Optional)
<ul>
<li>The text "Deep Learning: Foundations & Concepts" by Christopher Bishop & Hugh Bishop is recommended but not required.
The free-to-use online version is at <a href="https://www.bishopbook.com/">Bishop Book</a> 
  </li>
  <li>
    Dive into Deep Learning <a href="https://d2l.ai/">D2LAI</a> is an excellent interactive online textbook and set of resources for Deep Learning ! (a PDF version of the entire book is also available online)
  </li>
</ul>

## Lectures
Lectures are Tuesdays and Thursdays, 6:30PM - 8PM, in 10 Evans or online
via Zoom. Lecture slides are provided via this website, and lecture videos are
provided via the bCourses ["Media Gallery"](https://bcourses.berkeley.edu/courses/1538180/external_tools/90481). Students are responsible for all lecture content.

This Ed post ["Lecture Schedule"](https://edstem.org/us/courses/64085/discussion/5196727) contains more info about the lecture schedule, including: location (eg 10 Evans vs online Zoom) and lecture recording links.

Here is an **optional weekly reading list** of supplemental material: [link](https://docs.google.com/document/d/1asBPDgxrI47hiQ2rjGBmdAXLWv02kopqDgcE4yyW718/edit). 
While this is not required for the course, we believe that the material here can enhance understanding of the course and, more broadly, gain further exposure to the DNN field.

### Lecture Slides
<ul>
<li>Lecture 01 [Week 1, 2024/08/29] <a href="assets/lecture_slides/data182_Lecture1_Introduction.pdf">Introduction</a> </li>
  <ul>
 <li> Bishop Book: Chapter 1 </li>
  </ul>
<li>Lecture 02 [Week 2, 2024/09/03] <a href="assets/lecture_slides/data182_Lecture2_MLReview_1.pdf"> ML Review Part 1 </a> </li>
 <ul>
  <li>Bishop Book: Chapter 2 </li>
 <li>Reading:  <a href="assets/readings/NLL.pdf"> Binary classification & logistic regression </a> </li>
 </ul>
<li> Lecture 03 [Week 2, 2024/09/05] <a href="assets/lecture_slides/data182_Lecture03_MLReview_2.pdf"> ML Review Part 2</a> </li>
<li> Lecture 04 [Week 3, 2024/09/10] <a href="assets/lecture_slides/data182_Lecture04_Neural_Networks.pdf">Neural Networks</a> </li>
<li>Lecture 05 [Week 3, 2024/09/12] <a href="assets/lecture_slides/data182_Lecture5_Optimization.pdf"> Optimization </a> </li>
 <ul>
  <li>Bishop Book: Chapter 7 </li>
 </ul>
<li> Lecture 06 [Week 4, 2024/09/17] <a href="assets/lecture_slides/data182_Lecture06_Building_Blocks.pdf">Building Blocks</a>
    <ul>
    <li>Python demo code: 
<a href="assets/lecture_slides/python/lecture06/tensor_demo.py">[tensor_demo.py]</a> 
<a href="assets/lecture_slides/python/lecture06/two_layer_linear_nn_demo.py">[two_layer_linear_nn_demo.py]</a>
<a href="assets/lecture_slides/python/lecture06/normalization_motivation.py">[normalization_motivation.py]</a>
<a href="assets/lecture_slides/python/lecture06/batch_norm_demo.py">[batch_norm_demo.py]</a>
<a href="assets/lecture_slides/python/lecture06/layer_norm_demo.py">[layer_norm_demo.py]</a>
<a href="assets/lecture_slides/python/lecture06/two_layer_nn_bells_and_whistles.py">[two_layer_nn_bells_and_whistles.py]</a>
    </li>
    </ul>
</li>
  <li>Lecture 07 [Week 4, 2024/09/19] <a href="assets/lecture_slides/data182_Lecture7_ConvolutionalNetworks.pdf"> Convolutional Neural Networks ("ConvNets") </a> </li>
 <ul>
  <li>Bishop Book: Chapter 10 </li>
 </ul>
<li> Lecture 08 [Week 5, 2024/09/24] <a href="assets/lecture_slides/data182_Lecture08_Backprop_Part2.pdf">Backprop Part 2</a> </li>
<li> Lecture 09 [Week 5, 2024/09/26] <a href="assets/lecture_slides/data182_Lecture9_RNNs.pdf"> Recurrent Neural Networks ("RNNs") </a> </li>
<li> Lecture 10 [Week 06, 2024/10/01] <a href="assets/lecture_slides/data182_Lecture10_Computer_Vision.pdf"> Computer Vision </a> </li>
<li> Lecture 11 [Week 06, 2024/10/03] <a href="assets/lecture_slides/data182_Lecture11_Transformers1.pdf"> Transformers - 1 </a> </li>
  
</ul>

[//]: # (- [2022/01/31: Neural Networks]&#40;/assets/lecture_slides/2022.01.31-neural-networks.pdf&#41;)

[//]: # (- [2022/02/07: Optimization]&#40;/assets/lecture_slides/2022.02.07-optimization.pdf&#41;)

[//]: # (- [2022/02/09: Building Blocks]&#40;/assets/lecture_slides/2022.02.09-building-blocks.pdf&#41;)

[//]: # (- [2022/02/14: ConvNets]&#40;/assets/lecture_slides/2022.02.14-conv-nets.pdf&#41;)

[//]: # (- [2022/02/23: RNNs]&#40;/assets/lecture_slides/2022.02.23-rnns.pdf&#41;)

[//]: # (- [2022/02/28: MT1 Review]&#40;/assets/lecture_slides/2022.02.28-mt1-review.pdf&#41;)

[//]: # (- [2022/03/07: Transformers Part 1]&#40;/assets/lecture_slides/2022.03.07-transformers-pt1.pdf&#41;)

[//]: # (- [2022/03/09: Transformers Part 2]&#40;/assets/lecture_slides/2022.03.09-transformers-pt2.pdf&#41;)

[//]: # (- [2022/03/14: Sequence to Sequence Models]&#40;/assets/lecture_slides/2022.03.14-seq2seq.pdf&#41;)

[//]: # (- [2022/03/28: Distribution Shift]&#40;/assets/lecture_slides/2022.03.28-distribution-shift.pdf&#41;)

[//]: # (- [2022/03/30: Robustness]&#40;/assets/lecture_slides/2022.03.30-robustness.pdf&#41;)

[//]: # (- [2022/04/04: Adversarial Examples]&#40;/assets/lecture_slides/2022.04.04-adversarial-examples.pdf&#41;)

[//]: # (- [2022/04/06: Generative Models]&#40;/assets/lecture_slides/2022.04.06-generative-models.pdf&#41;)

[//]: # (- [2022/04/11: Self-Supervised Learning]&#40;/assets/lecture_slides/2022.04.11-self-supervised.pdf&#41;)

[//]: # (- [2022/04/13: Massive Models]&#40;/assets/lecture_slides/2022.04.13-massive-models.pdf&#41;)

[//]: # (- [2022/04/25: MT2 Review]&#40;/assets/lecture_slides/2022.04.25-mt2-review.pdf&#41;)

## Discussion Sections
The discussion sections will not cover new material, but rather will give you
additional practice solving problems. You can attend any discussion section you
like. However, if there are less crowded sections that fit your schedule, those
offer more opportunities for you to interact with your TA.

### Section Notes

- Discussion 01 (Week 3): [Section Notes](/assets/section_notes/week03.pdf), [Solution](/assets/section_notes/week03_solution.pdf)

- Discussion 02 (Week 4): [Section Notes](/assets/section_notes/week04.pdf), [Solution](/assets/section_notes/week04_solution.pdf)

- Discussion 03 (Week 5): [Section Notes](/assets/section_notes/week05.pdf), [Solution](/assets/section_notes/week05_solution.pdf). [Notebook](/assets/section_notes/DataC182_DiscussionWorksheet_3.ipynb). Open notebook on [Google Colab](https://colab.research.google.com/github/datac182fa24/datac182fa24.github.io/blob/main/assets/section_notes/DataC182_DiscussionWorksheet_3.ipynb)

- Discussion 04 (Week 6): [Section Notes](/assets/section_notes/week06.pdf), [Solution](/assets/section_notes/week06_solution.pdf)

- Discussion 05 (Week 7): [Section Notes](/assets/section_notes/week07.pdf);, [Solution](/assets/section_notes/week07_solution.pdf)

[//]: # (- Discussion 6: [Section Notes]&#40;/assets/section_notes/week6.pdf&#41;, [Solution]&#40;/assets/section_notes/week6_solution.pdf&#41;)

[//]: # (- Discussion 7: [Section Notes]&#40;/assets/section_notes/week7.pdf&#41;, [Solution]&#40;/assets/section_notes/week7_solution.pdf&#41;)

- Discussion 08 (Week 11): [Section Notes](/assets/section_notes/week08.pdf);, [Solution](/assets/section_notes/week08_solution.pdf)

[//]: # (- Discussion 9: [Section Notes]&#40;/assets/section_notes/week9.pdf&#41;, [Solution]&#40;/assets/section_notes/week9_solution.pdf&#41;)

[//]: # (- Discussion 10: [Section Notes]&#40;/assets/section_notes/week10.pdf&#41;, [Solution]&#40;/assets/section_notes/week10_solution.pdf&#41;)

[//]: # (- Discussion 11: [Section Notes]&#40;/assets/section_notes/week11.pdf&#41;, [Solution]&#40;/assets/section_notes/week11_solution.pdf&#41;)

[//]: # (- Discussion 12: [Section Notes]&#40;/assets/section_notes/week12.pdf&#41;, [Solution]&#40;/assets/section_notes/week12_solution.pdf&#41;)

## Homeworks
All homeworks are graded for accuracy.
See the [course syllabus](https://datac182fa24.github.io/syllabus/#homeworks) for info about collaboration, slip day, late policy.

### Homeworks

- [Homework 1](https://github.com/datac182fa24/datac182_hw1_student), **due Tues Oct 8th 2024 11:59 PM PST**

- [Homework 2](/assets/homeworks/DataC182_FA24_HW2-2.pdf), **due Sun Oct 27th 2024 11:59 PM PST**

[//]: # (- [Homework 3]&#40;https://github.com/cs182sp22/cs182_hw3_student&#41;, **due 04/11/2022 11:59 PM PST**)

[//]: # (- [Homework 4]&#40;https://colab.research.google.com/drive/1lLHAuIs4YW2tmG8Mhbc7VgvM7oiFev0E?usp=sharing&#41;, **due 05/02/2022 11:59 PM PST**)
