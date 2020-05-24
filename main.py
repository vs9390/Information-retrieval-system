from inverted_index import InvertedIndex

# Building instance of index class
inverted_index_instance = InvertedIndex()

# Calling search function for every query

inverted_index_instance.lookup_using_cosine_similarity("Seizing the opportunity, Booth crept up from behind and at about 10:13 pm, aimed at the back of Lincoln's head and fired at point-blank range, mortally wounding the President")
#
inverted_index_instance.lookup_using_cosine_similarity('Along the shores of the Caspian Sea it is temperate, while the higher mountain elevations are generally cold.')
#
inverted_index_instance.lookup_using_cosine_similarity("Agassi, along with five athlete partners (including Wayne Gretzky, Joe Montana, Shaquille O'Neal, Ken Griffey, Jr., and Monica Seles) opened a chain of sports-themed restaurant named Official All Star Café in 1996")
#
inverted_index_instance.lookup_using_cosine_similarity('One of the attributes of ANOVA which ensured its early popularity was computational elegance. The structure of the additive model allows solution for the additive coefficients by simple algebra rather than by matrix calculations')
#
inverted_index_instance.lookup_using_cosine_similarity('Many critics doubted the viability of four-wheel drive racers, thinking them to be too heavy and complex, yet the Quattro was to become a successful car.')