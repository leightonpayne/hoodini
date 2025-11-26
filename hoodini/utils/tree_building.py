def calculate_taxid_distances(taxids, update_db=False):
    """
    Calculate pairwise distances between given taxonomic IDs using ete3 by passing node objects.

    Parameters:
    - taxids (list): A list of taxonomic IDs (integers or strings).
    - update_db (bool): Whether to update the local NCBI taxonomy database. Default is False.

    Returns:
    - dict: A dictionary with keys as tuples of taxid pairs and values as their distances.

    Raises:
    - ValueError: If any of the taxids are missing from the taxonomy database.
    """
    
    # Ensure taxids are strings for consistency with ete3 node names
    taxids_str = [str(int(taxid)) for taxid in taxids]
    
    # Initialize NCBITaxa object
    ncbi = ete3.NCBITaxa()
    
    # Optional: Update the taxonomy database
    if update_db:
        try:
            print("Updating NCBI taxonomy database. This may take a while...")
            ncbi.update_taxonomy_database()
            print("Taxonomy database updated successfully.")
        except Exception as e:
            print(f"Error updating taxonomy database: {e}")
            sys.exit(1)
    
    # Retrieve the phylogenetic tree topology for the given taxids
    try:
        tree = ncbi.get_topology(taxids_str, intermediate_nodes=True)
    except Exception as e:
        print(f"Error retrieving topology: {e}")
        sys.exit(1)
    
    # Debug: Collect all taxids present in the tree
    tree_taxids = set()
    taxid_to_node = {}
    for node in tree.traverse():
        try:
            taxid = int(node.name)
            tree_taxids.add(taxid)
            taxid_to_node[taxid] = node
        except ValueError:
            # node.name might not be a taxid (e.g., common ancestor labels)
            continue
    
    # Convert input taxids to integers for comparison
    try:
        taxids_int = [int(taxid) for taxid in taxids_str]
    except ValueError as ve:
        print(f"Error converting taxids to integers: {ve}")
        sys.exit(1)
    
    # Identify missing taxids
    missing_taxids = [taxid for taxid in taxids_int if taxid not in tree_taxids]
    if missing_taxids:
        print("The following taxids are missing from the taxonomy tree:")
        for mtaxid in missing_taxids:
            # Fetch the scientific name for better clarity
            try:
                name = ncbi.get_taxid_translator([mtaxid]).get(mtaxid, "Unknown")
            except Exception as e:
                name = "Unknown"
            print(f" - {mtaxid} ({name})")
        raise ValueError("Some taxids are missing from the taxonomy tree. Please verify their validity.")
    else:
        print("All taxids are present in the taxonomy tree.")
    
    # Initialize a dictionary to store pairwise distances
    distances = {}
    
    # Iterate over all unique pairs of taxids using combinations
    print("\nCalculating pairwise distances using node objects...")
    for taxid1, taxid2 in itertools.combinations(taxids_int, 2):
        try:
            node1 = taxid_to_node[taxid1]
            node2 = taxid_to_node[taxid2]
            distance = tree.get_distance(node1, node2)
            distances[(taxid1, taxid2)] = distance
        except Exception as e:
            print(f"Error calculating distance between {taxid1} and {taxid2}: {e}")
    
    # Check if any distances were calculated
    if not distances:
        raise ValueError("No distances were calculated. Please check the taxids and try again.")
    
    print("\nPairwise Distances Calculated Successfully.")
    
    return distances