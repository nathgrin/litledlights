def cycle_individual_lights(color:tuple[int]=(255,255,255),
        dt: float=1., loop: bool=True,
        off_color:tuple[int]=(0,0,0)):

    with get_strip() as strip:
        strip.fill( off_color )
        while True:

            for i in range(len(strip)):
                strip[i-1] = off_color
                strip[i]   = color
                strip.show()
                time.sleep(dt)
            if not loop:
                break

    
