class IPGXMLFile:
    def __init__(self, file_path):
        """
        Initialize the IPGXMLFile object with the path to an XML file and parse it.

        Args:
            file_path (str): The path to the XML file (IPG, elink, efetch, or assembly).
        """
        self.file_path = file_path
        self.xml_tree = ET.parse(self.file_path)
        self.root = self.xml_tree.getroot()

    def to_dict(self, mode):
        """
        Converts the XML element tree into a dictionary based on the mode.

        Args:
            mode (str): 'ipg', 'nuc2ass', 'nucsum', or 'asssum' for assembly summary parsing.

        Returns:
            dict: A dictionary representation of the XML data.
        """
        assert mode in ["ipg", "nuc2ass", "nucsum", "asssum"]
        parsed_dict = {}

        if mode == "ipg":
            # Parsing IPGReport data
            for ipg_report in self.root.findall('IPGReport'):
                ipg_id = ipg_report.get('ipg')
                product_acc = ipg_report.get('product_acc')

                product = ipg_report.find('Product')
                product_details = product.attrib if product is not None else {}

                proteins = []
                for protein in ipg_report.findall('.//Protein'):
                    protein_accver = protein.get('accver')
                    protein_details = {k: v for k, v in protein.attrib.items() if k != 'accver'}

                    cds_list = []
                    for cds in protein.findall('.//CDS'):
                        cds_info = cds.attrib
                        cds_list.append(cds_info)

                    protein_info = {
                        'protein_accver': protein_accver,
                        'protein_details': protein_details,
                        'cds_list': cds_list
                    }
                    proteins.append(protein_info)

                parsed_dict[ipg_id] = {
                    'product_acc': product_acc,
                    'product_details': product_details,
                    'proteins': proteins
                }

        elif mode == "nuc2ass":
            # Parsing elink XML structure (nuc2ass)
            for link_set in self.root.findall('LinkSet'):
                db_from = link_set.findtext('DbFrom')
                id_list = [id_tag.text for id_tag in link_set.findall('.//IdList/Id')]

                link_set_db = link_set.find('LinkSetDb')
                if link_set_db is not None:
                    db_to = link_set_db.findtext('DbTo')
                    link_name = link_set_db.findtext('LinkName')
                    linked_ids = [link.findtext('Id') for link in link_set_db.findall('.//Link')]
                else:
                    db_to = link_name = None
                    linked_ids = []

                parsed_dict[tuple(id_list)] = {
                    'db_from': db_from,
                    'db_to': db_to,
                    'link_name': link_name,
                    'linked_ids': linked_ids
                }

        elif mode == "nucsum":
            # Parsing efetch summary XML structure (nucsum)
            for doc_sum in self.root.findall('DocSum'):
                doc_id = doc_sum.findtext('Id')
                doc_items = {item.get('Name'): item.text for item in doc_sum.findall('Item')}
                parsed_dict[doc_id] = doc_items

        elif mode == "asssum":
            # Parsing assembly summary data
            for doc_summary in self.root.findall('.//DocumentSummary'):
                uid = doc_summary.get('uid')
                entry = {child.tag: child.text for child in doc_summary}
                parsed_dict[uid] = entry

        return parsed_dict

    def to_dataframe(self, mode):
        """
        Converts the XML data into a pandas DataFrame.

        Args:
            mode (str): 'ipg', 'nuc2ass', 'nucsum', or 'asssum' for assembly summary parsing.

        Returns:
            DataFrame: A pandas DataFrame representing the parsed data.
        """
        parsed_dict = self.to_dict(mode)
        flattened_data = []

        if mode == "ipg":
            # Flatten IPG data
            for ipg_id, details in parsed_dict.items():
                base_info = {'ipg_id': ipg_id, 'product_acc': details['product_acc']}
                base_info.update(details['product_details'])

                for protein in details['proteins']:
                    protein_info = protein['protein_details'].copy()
                    protein_info['protein_accver'] = protein['protein_accver']

                    if protein['cds_list']:
                        for cds in protein['cds_list']:
                            row = {**base_info, **protein_info, **cds}
                            flattened_data.append(row)
                    else:
                        row = {**base_info, **protein_info}
                        flattened_data.append(row)

        elif mode == "nuc2ass":
            # Flatten elink data
            for ids, details in parsed_dict.items():
                id_list_str = ','.join(ids)
                for linked_id in details['linked_ids']:
                    row = {
                        'db_from': details['db_from'],
                        'id_list': id_list_str,
                        'db_to': details['db_to'],
                        'link_name': details['link_name'],
                        'linked_id': linked_id
                    }
                    flattened_data.append(row)

        elif mode == "nucsum":
            # Flatten efetch summary data
            for doc_id, doc_items in parsed_dict.items():
                row = {'doc_id': doc_id}
                row.update(doc_items)
                flattened_data.append(row)

        elif mode == "asssum":
            # Flatten assembly summary data
            for uid, details in parsed_dict.items():
                row = {'uid': uid}
                row.update(details)
                flattened_data.append(row)

        return pd.DataFrame(flattened_data)