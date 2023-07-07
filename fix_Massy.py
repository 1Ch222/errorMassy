import os
from PIL import Image

def process_images(folder_path):
    # Parcourir tous les fichiers du dossier
    for filename in os.listdir(folder_path):
        # Vérifier si le fichier est au format PNG et ne se termine pas par "labelIds.png"
        if filename.lower().endswith(".png") and not filename.lower().endswith("labelids.png"):
            file_path = os.path.join(folder_path, filename)
            process_image(file_path)

def process_image(file_path):
    # Ouvrir l'image
    image = Image.open(file_path)

    # Convertir l'image en noir et blanc
    black_white_image = convert_to_black_white(image)

    # Obtenir les pixels par particule triés par ordre de taille décroissante
    pixels_per_particle = count_pixels_per_particle(black_white_image)

    # Obtenir la taille en pixels de la plus grande particule blanche
    if len(pixels_per_particle) > 0:
        particle_id, particle_pixels = pixels_per_particle[0]
        particle_size = len(particle_pixels)
        print(f"Taille de la plus grande particule blanche dans {file_path}: {particle_size} pixels")

def convert_to_black_white(image):
    # Convertir l'image en mode RVB
    rgb_image = image.convert("RGB")

    # Obtenir les dimensions de l'image
    width, height = rgb_image.size

    # Créer une nouvelle image en noir et blanc
    black_white_image = Image.new("1", (width, height))

    # Parcourir tous les pixels de l'image
    for y in range(height):
        for x in range(width):
            # Obtenir la couleur du pixel (R, G, B)
            r, g, b = rgb_image.getpixel((x, y))

            # Vérifier si le pixel est blanc
            if r == 224 and g == 172 and b == 51:
                # Le pixel est blanc, le marquer comme blanc dans l'image en noir et blanc
                black_white_image.putpixel((x, y), 1)
            else:
                # Le pixel est noir, le marquer comme noir dans l'image en noir et blanc
                black_white_image.putpixel((x, y), 0)

    return black_white_image

def count_pixels_per_particle(image):
    # Convertir l'image en mode RVB
    rgb_image = image.convert("RGB")

    # Obtenir les dimensions de l'image
    width, height = rgb_image.size

    # Créer une image masque pour marquer les pixels déjà comptés
    mask = Image.new("1", (width, height))

    # Initialiser un dictionnaire pour stocker les pixels par particule
    pixels_per_particle = {}

    # Parcourir tous les pixels de l'image
    for y in range(height):
        for x in range(width):
            # Obtenir la couleur du pixel (R, G, B)
            r, g, b = rgb_image.getpixel((x, y))

            # Vérifier si le pixel est blanc et n'a pas été compté
            if r == 255 and g == 255 and b == 255 and mask.getpixel((x, y)) == 0:
                # Trouver tous les pixels de la particule en utilisant le remplissage par diffusion
                particle_pixels = flood_fill(rgb_image, mask, x, y)

                # Assigner les pixels à une nouvelle particule
                particle_id = len(pixels_per_particle) + 1
                pixels_per_particle[particle_id] = particle_pixels

    # Trier les pixels par particule par ordre de taille décroissante
    sorted_pixels_per_particle = sorted(pixels_per_particle.items(), key=lambda item: len(item[1]), reverse=True)

    return sorted_pixels_per_particle

def flood_fill(image, mask, start_x, start_y):
    width, height = image.size

    stack = [(start_x, start_y)]
    visited = set()
    pixels = []

    while stack:
        x, y = stack.pop()

        if (x, y) in visited:
            continue
        visited.add((x, y))

        if mask.getpixel((x, y)) == 0:
            r, g, b = image.getpixel((x, y))
            if r == 255 and g == 255 and b == 255:
                pixels.append((x, y))
                mask.putpixel((x, y), 1)

                # Ajouter les pixels voisins dans la pile
                if x > 0:
                    stack.append((x - 1, y))
                if x < width - 1:
                    stack.append((x + 1, y))
                if y > 0:
                    stack.append((x, y - 1))
                if y < height - 1:
                    stack.append((x, y + 1))

    return pixels

# Exemple d'utilisation
folder_path = "/home/poc2014/dataset/temp/INFRA10/semantic_segmentation_truth/val/Massy/"

process_images(folder_path)
