from utils import get_strip
import time

def plane(strip=None,
        color:tuple[int]=(255,255,255),
        dt: float=1., loop: bool=False,
        off_color:tuple[int]=(0,0,0)):
    
    strip = get_strip() if strip is None else strip
    

def cycle_sequential(strip=None,
        color:tuple[int]=(255,255,255),
        dt: float=1., loop: bool=False,
        off_color:tuple[int]=(0,0,0)):
    
    strip = get_strip() if strip is None else strip
    
    with strip:
        strip.fill( off_color )
        strip.show()
        while True:

            for l in range(len(strip)):
                strip[l-1] = off_color
                strip[l]   = color
                strip.show()
                time.sleep(dt)
            if not loop:
                break


def blink_binary(inlist: list[int],nbits: int=None,
        strip=None,
        color:tuple[int]=(255,255,255),
        dt: float=1., loop: int=False,
        off_color:tuple[int]=(0,0,0)):
    
    strip = get_strip() if strip is None else strip
    
    
    bins = [format(n, 'b') for n in inlist]
    lens = [ len(n) for n in bins ]
    nbits = max(lens) if nbits is None else max(max(lens),nbits) # Watch out, theres no warning here!
    bins = [ x.zfill(nbits) for x in bins ]
    # ~ print(bins)
    # ~ print(lens)
    
    with strip:
        while True:
            strip.fill( off_color )
            strip.show()
            
            for i in range(nbits):
                for l in range(len(strip)):
                    strip[l] = color if int(bins[l][i]) else off_color
                
                strip.show()
                time.sleep(dt)
                
            if not loop:
                break
        
        
        
        

    
def main():
    strip = get_strip()
    inlist = range(len(strip))
    blink_binary(inlist,strip=strip)
    

if __name__ == "__main__":
    main()
    
