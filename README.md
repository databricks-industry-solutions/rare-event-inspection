![image](https://user-images.githubusercontent.com/86326159/206014015-a70e3581-e15c-4a10-95ef-36fd5a560717.png)

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

*Kakapo (KAH-kə-poh) implements a standard set of APIs for outlier detection at scale on Databricks. It provides an integration of the vast PyOD library of outlier detection algorithms with MLFlow for tracking and packaging of models and hyperopt for exploring vast, complex and heterogeneous search spaces.*

Kakapo aims to address a number of considerations to ensure robust solutions:
- Future proofing and scalability, i.e. how to handle not just today’s workloads but have a framework that scales as requirements change - e.g. volume/ velocity/ complexity increases
- Productivity and collaboration, i.e. how to ensure that work and ideas can be easily shared
- Governance and auditability, i.e. how to can collect and log metadata, ensure robust audit trails and ultimately produce data that can be trusted
___
<iliya.kostov@databricks.com> \
<milos.colic@databricks.com>

___


&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| PyOD                                 | A Comprehensive and Scalable Python Library for Outlier Detection (Anomaly Detection)      | BSD        | https://pypi.org/project/pyod                      |
| EMMV                                 | Metrics for unsupervised anomaly detection models      | MIT        | https://pypi.org/project/emmv                      |

## Getting started

Although specific solutions can be downloaded as .dbc archives from our websites, we recommend cloning these repositories onto your databricks environment. Not only will you get access to latest code, but you will be part of a community of experts driving industry best practices and re-usable solutions, influencing our respective industries. 

<img width="500" alt="add_repo" src="https://user-images.githubusercontent.com/4445837/177207338-65135b10-8ccc-4d17-be21-09416c861a76.png">

To start using this solution in Databricks simply follow these steps: 

1. Clone the repository in Databricks using [Databricks Repos](https://www.databricks.com/product/repos)
2. Attach the `01_Kakapo_ walkthrough` notebook to any ML runtime cluster and execute the notebook via Run-All.
3. You might want to modify the samples in the solution to your need, collaborate with other users and run the code samples against your own data. To do so start by changing the Git remote of your repository  to your organization’s repository vs using our samples repository (learn more). You can now commit and push code, collaborate with other user’s via Git and follow your organization’s processes for code development.

The cost associated with running the solution is the user's responsibility.


## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 
