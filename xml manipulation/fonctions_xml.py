import xml.etree.ElementTree as ET

import numpy as np


def get_annots_from_xml(path):
    """
    Récupère les annotations contenues dans un fichier .xml au format Pascal VOC.
    Retourne un array utilisable en entrée des modèles Tensorflow et des fonctions d'affichage.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    
    size_x = int(root.findall('size')[0][0].text)
    size_y = int(root.findall('size')[0][1].text)
    objects_annot = root.findall('object')
    arr_box_annot = np.zeros((len(objects_annot), 4), dtype=np.float32)

    for i, box_annot in enumerate(root.findall('object')):
        arr_box_annot[i, :] = [int(box_annot[4][1].text) / size_y,
                               int(box_annot[4][0].text) / size_x,
                               int(box_annot[4][3].text) / size_y,
                               int(box_annot[4][2].text) / size_x]
    return arr_box_annot

def create_xml_from_detection(path_xml, detection_boxes, detection_scores,
                              seuil_detection, name_image, path_image, width, height, depth):
    """
    Génère un XML au format Pascal VOC contenant les annotations détectées avec une confiance supérieure au seuil.
    """
    # Création de la racine du document et des infos générales
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "patchs"
    ET.SubElement(root, "filename").text = str(name_image)
    ET.SubElement(root, "path").text = str(path_image)
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Detection"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)
    ET.SubElement(root, "segmented").text = "0"
    
    # Nombre de bounding boxes détectées avec assez de confiance
    nb_boxes = np.sum(detection_scores >= seuil_detection)
    
    # Création de chaque bounding box
    # Le modèle renvoie les bounding boxes par ordre décroissant de confiance
    for i in range(nb_boxes):
        arr_boxes = np.array(detection_boxes)[i, :]
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "chardon"
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(np.round(arr_boxes[1] * width)))
        ET.SubElement(bndbox, "ymin").text = str(int(np.round(arr_boxes[0] * height)))
        ET.SubElement(bndbox, "xmax").text = str(int(np.round(arr_boxes[3] * width)))
        ET.SubElement(bndbox, "ymax").text = str(int(np.round(arr_boxes[2] * height)))
                                                    
    # Assemblage de l'arbre
    tree = ET.ElementTree(root)
    
    # Ecriture du fichier
    tree.write(path_xml)
