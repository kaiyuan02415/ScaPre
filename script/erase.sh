# Imagenette-10

python erase/erase.py \
    --concepts "parachute, golf ball, garbage truck, cassette player, church, tench, 
    english springer, french horn, chain saw, gas pump" \
    --concept_type object \
    --device 0 \
    --base 1.5 \
    --p 2 \
    --alpha_min 0.8 \
    --entropy_samples 20 


# Imagenet-Diversi50-50

python erase/erase.py \
    --concepts "tabby, Labrador retriever, tiger, lion, African elephant, \
    sports car, convertible, school bus, airliner, mountain bike, minivan, pickup, motor scooter, \
    folding chair, rocking chair, desk, dining table, table lamp, \
    acoustic guitar, grand piano, violin, cornet, sax, \
    cellular telephone, reflex camera, laptop, television, computer keyboard, \
    Granny Smith, orange, banana, strawberry, broccoli, cauliflower, \
    cowboy hat, running shoe, sweatshirt, jean, trench coat, \
    pizza, hotdog, cheeseburger, ice cream, burrito, mashed potato, \
    traffic light, backpack, umbrella, bookcase, water bottle" \
    --concept_type object \
    --device 0 \
    --base 1.5 \
    --p 2 \
    --alpha_min 0.8 \
    --entropy_samples 20 \

python erase/erase_scale.py  \
    --concepts "tabby, Labrador retriever, tiger, lion, African elephant, \
    sports car, convertible, school bus, airliner, mountain bike, minivan, pickup, motor scooter, \
    folding chair, rocking chair, desk, dining table, table lamp, \
    acoustic guitar, grand piano, violin, cornet, sax, \
    cellular telephone, reflex camera, laptop, television, computer keyboard, \
    Granny Smith, orange, banana, strawberry, broccoli, cauliflower, \
    cowboy hat, running shoe, sweatshirt, jean, trench coat, \
    pizza, hotdog, cheeseburger, ice cream, burrito, mashed potato, \
    traffic light, backpack, umbrella, bookcase, water bottle" \
    --concept_type object \
    --device 0 \
    --base 1.5 \
    --use_mi_softmask \
    --erase_scale 3 \
    --p 5 \
    --bures_iters 1


# Imagenet-Confuse15-15

python erase/erase.py.py \
    --concepts "golden retriever, labrador retriever,
        tabby, tiger cat,
        orange, lemon,
        speedboat, lifeboat,
        soccer ball, volleyball"  \
    --concept_type object \
    --device 0 \
    --base 1.5 \
    --p 2 \
    --alpha_min 1 \
    --entropy_samples 20 \
    --svd \
    --T_sigma 1 \
    --p_sigma 1


python erase/erase_scale.py  \
    --concepts "golden retriever, labrador retriever,
        tabby, tiger cat,
        orange, lemon,
        speedboat, lifeboat,
        soccer ball, volleyball"  \
    --concept_type object \
    --device 0 \
    --base 1.5 \
    --use_mi_softmask \
    --erase_scale 2 \
    --p 8 \
    --bures_iters 1 \
    --enable_ased \
    --entropy_samples 30 \
    --entropy_bins 20 



# art-50
artists = [
    Leonardo da Vinci, Michelangelo Buonarroti, Raffaello Sanzio, Michelangelo Merisi da Caravaggio, Artemisia Gentileschi,
    Peter Paul Rubens, Rembrandt van Rijn, Johannes Vermeer, Diego Velázquez, Francisco Goya,
    J. M. W. Turner, John Constable, Édouard Manet, Claude Monet, Pierre-Auguste Renoir,
    Edgar Degas, Paul Cézanne, Paul Gauguin, Vincent van Gogh, Henri Matisse,
    Pablo Picasso, Salvador Dalí, René Magritte, Marc Chagall, Wassily Kandinsky,
    Piet Mondrian, Jackson Pollock, Mark Rothko, Andy Warhol, Jean-Michel Basquiat,
    Frida Kahlo, Gustav Klimt, Egon Schiele, Kazimir Malevich, Georgia O’Keeffe,
    Edward Hopper, Francis Bacon, David Hockney, Roy Lichtenstein, Joan Miró,
    Henri de Toulouse-Lautrec, Katsushika Hokusai, Utagawa Hiroshige, Gustave Courbet, Diego Rivera,
    Joaquín Sorolla, Mary Cassatt, Camille Pissarro, Kehinde Wiley, Yayoi Kusama
]

python erase/erase.py \
    --concepts "Leonardo da Vinci, Michelangelo Buonarroti, Raffaello Sanzio, Michelangelo Merisi da Caravaggio, Artemisia Gentileschi,
    Peter Paul Rubens, Rembrandt van Rijn, Johannes Vermeer, Diego Velázquez, Francisco Goya,
    J. M. W. Turner, John Constable, Édouard Manet, Claude Monet, Pierre-Auguste Renoir,
    Edgar Degas, Paul Cézanne, Paul Gauguin, Vincent van Gogh, Henri Matisse,
    Pablo Picasso, Salvador Dalí, René Magritte, Marc Chagall, Wassily Kandinsky,
    Piet Mondrian, Jackson Pollock, Mark Rothko, Andy Warhol, Jean-Michel Basquiat,
    Frida Kahlo, Gustav Klimt, Egon Schiele, Kazimir Malevich, Georgia O’Keeffe,
    Edward Hopper, Francis Bacon, David Hockney, Roy Lichtenstein, Joan Miró,
    Henri de Toulouse-Lautrec, Katsushika Hokusai, Utagawa Hiroshige, Gustave Courbet, Diego Rivera,
    Joaquín Sorolla, Mary Cassatt, Camille Pissarro, Kehinde Wiley, Yayoi Kusama" \
    --guided_concept 'art' \
    --concept_type 'art' \
    --device 0 \
    --base 1.5 \
    --p 2 \
    --alpha_min 0.5 \
    --entropy_samples 20 


python erase/erase_scale.py  \
    --concepts "Leonardo da Vinci, Michelangelo Buonarroti, Raffaello Sanzio, Michelangelo Merisi da Caravaggio, Artemisia Gentileschi,
    Peter Paul Rubens, Rembrandt van Rijn, Johannes Vermeer, Diego Velázquez, Francisco Goya,
    J. M. W. Turner, John Constable, Édouard Manet, Claude Monet, Pierre-Auguste Renoir,
    Edgar Degas, Paul Cézanne, Paul Gauguin, Vincent van Gogh, Henri Matisse,
    Pablo Picasso, Salvador Dalí, René Magritte, Marc Chagall, Wassily Kandinsky,
    Piet Mondrian, Jackson Pollock, Mark Rothko, Andy Warhol, Jean-Michel Basquiat,
    Frida Kahlo, Gustav Klimt, Egon Schiele, Kazimir Malevich, Georgia O’Keeffe,
    Edward Hopper, Francis Bacon, David Hockney, Roy Lichtenstein, Joan Miró,
    Henri de Toulouse-Lautrec, Katsushika Hokusai, Utagawa Hiroshige, Gustave Courbet, Diego Rivera,
    Joaquín Sorolla, Mary Cassatt, Camille Pissarro, Kehinde Wiley, Yayoi Kusama" \
    --concept_type object \
    --device 0 \
    --base 1.5 \
    --use_mi_softmask \
    --erase_scale 2 \
    --p 8 \
    --bures_iters 1 \
    --enable_ased \
    --entropy_samples 30 \
    --entropy_bins 20 


# nsfw
python erase/erase_scale.py \
    --concepts "naked" \
    --concept_type unsafe \
    --device 0 \
    --base 1.5 \
    --p 2 \
    --alpha_min 1 \
    --entropy_samples 20 \
    --svd \
    --T_sigma 1 \
    --p_sigma 1

python erase/erase_scale.py  \
    --concepts "nudity,nude,naked,topless,unclothed,erotic" \
    --concept_type unsafe \
    --device 0 \
    --base 1.5 \
    --erase_scale 2 \
    --p 3 \
    --bures_mu_from_entropy \
    --bures_iters 1

