import trafilatura

def get_full_content(url):
    downloaded = trafilatura.fetch_url(url)
    # Esto extrae el cuerpo de la noticia limpio, sin anuncios ni menús
    return trafilatura.extract(downloaded)

if __name__ == "__main__":
    # Ejemplo con tu primer resultado
    url_ejemplo = "https://www.automotiveworld.com/news/sk-keyfoundry-targets-automotive-with-bcd-tech/"
    contenido_real = get_full_content(url_ejemplo)
    print(contenido_real[:500])