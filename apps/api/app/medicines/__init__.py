"""The Indian medicines catalogue Clinexa looks names up in.

Three sources, read from ``data/`` (app/medicines/catalog.py):

* the A to Z medicines dataset of India: about 250,000 branded products,
* the Jan Aushadhi (PMBJP) product list: about 2,100 generic medicines,
* the National List of Essential Medicines 2022: the essential generics, with the
  strengths and forms each comes in.

They are kept in Qdrant (app/medicines/store.py), one point for each medicine name. A name
is found by how it is spelt and how it sounds, not by what it means: "glycomate five
hundred" has to find "Glycomet 500". So each point carries a sparse vector of letter
groups and sound keys (app/medicines/text.py), which needs no language model.

The voice agent uses it when a caller names a medicine (app/medicines/lookup.py): the
catalogue's spelling, what the medicine contains, and its strength where there is only
one it could be, are what Clinexa is given to say back (app/tools/knowledge.py).
"""
