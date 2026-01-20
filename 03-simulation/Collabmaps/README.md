# Collabmaps
An extension of the Datamaps package to show links between locations

Jeff Drocco, drocco1@llnl.gov

Instructions:

1) Hover over a bubble to display collaborator connections and a panel displaying featured authors
   and papers.

2) Click once on any bubble to enter panel mode, to access links to the PubMed versions of the paper
   and Google searches of the top authors. Click on any bubble to exit panel mode.

Input:

Accepts a json file consisting of a list of institutions, each with the following elements:

name: string
country: ISO 3166-1 alpha-3 country code
TopAuthors: list of names [last,first]
latitude: float
longitude: float
connections: list of lat/long locations of linked bubbles
radius: int (radius of bubble)
address: string
TopPapers: list of publications [title,URL,PubMedURL,date]
fillKey: color, palette defined in collabmaps.world.js

