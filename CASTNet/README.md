
CASTNET: https://arxiv.org/abs/1905.04714 

----

cook-county-features and ma-features make appropriate input files to feed to CASTNet
--> these files are saved in 'Data'

is the notebook that runs CASTNet to get predictions, i would start there to explore code

Changes to our CASTNet from the original include:
- removing crime and replacing with SVI 
- we use annual level (instead of daily/weekly)
- we use census tracts 
- general updates due to outdated tensorflow code
- added 'batches' parameter for train() to give option if you want to run w/ batches or not

---

** If you need to reference original CASTNet code and files, see CASTNet-master