## REPLACE TEMP_IDS with unique_IDS in blast, deffinder, etc to avoid problems with IDs lenghts
## Add search button

update recipe padloc




# Add domains
# Add genomad as regions to GFF
# Add ncrnas as regions to GFF
# Add blast as regions to GFF
# Add CRISPR arrays as regions to GFF
# Compute wGRR too beyond AAI



Improve logs and beutify output

Check threads used in every tool

Check style and structure consistency of the modules

Fix problem of mislaignment of tree labels (looks like it is using the last computed one, not the current one, from textlabelby)

Switch dataframe backend to polars

Ensure homogeneous column naming across modules and reduce 

Add MANIAC to masure similarity

Add progressivemauve or mummer for alignments

Use the CIGAR of minimap to split the alignment into smaller chunks

Change name of variable mod from win_mod to win_mod, angit 
Choose between taxoinq and ete3