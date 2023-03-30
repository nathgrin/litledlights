
# litledlights

LitLedLights, they are Lit

## Resources

- adafruit
  - <https://learn.adafruit.com/neopixels-on-raspberry-pi/raspberry-pi-wiring>
  - <https://learn.adafruit.com/neopixels-on-raspberry-pi/python-usage>
  - [adafruit neopixel uberguide](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwisw86m_uz8AhVFM-wKHQVvAJgQFnoECA8QAQ&url=https%3A%2F%2Fcdn-learn.adafruit.com%2Fdownloads%2Fpdf%2Fadafruit-neopixel-uberguide.pdf&usg=AOvVaw1-UNr6xUSFV5fscJPYqsFR)
  - [adafruit circuitpython NeoPixel python source](https://github.com/adafruit/Adafruit_CircuitPython_NeoPixel/blob/main/neopixel.py)
- Colors
  - [rgb/hsv colorpicker tool](https://math.hws.edu/graphicsbook/demos/c2/rgb-hsv.html)
  - [stackoverflow python functions](https://stackoverflow.com/questions/24852345/hsv-to-rgb-color-conversion)

## To do

- coords2dto3d
  - fix whe nno mouse available
  - fix click dist ditribution window
  - fix change reference inds
  - fix iterative version
  - test on ledz
  - test on rpi
  - pff
- level converter "chip"
- robust wiring
  - Cables with more power throughput
- calibrate makecoords3d
  - clean up new_camera_mess in combine_2d_to_3d
  - build sampler for triangulation
- fix error after 'lll play blink_binary' exit with gives gpio.pin is Nonetype?

## Shopping list

- heatsinks

## Roadmap

- [x] setup rpi
- [x] link up stuff
- [ ] timing
- [ ] control leds
  - [x] get leds to light
  - [x] control leds
    - [x] Library (adafruit-circuitpython-neopixel)
  - [ ] make API
- [ ] coords
  - [ ] clean up sequential photography function
  - [x] setup camera
    - [x] live view camera?
    - [x] calibrate camera
  - [x] get pts from picture/video
    - [ ] improve get pts
    - [ ] combine pictures
    - [x] rotate coords
    - [ ] fix wrong coords
    - [x] fix missing coords
    - [ ] make 2d_to_3d accesible without mouse
    - [ ] iterate with missing coords
  - [x] coords class
  - [ ] coords module organisation
  - [x] save
  - [ ] share coords (gift?)
  - [ ] coordinate systems (spherical, cilindrical)
  - [ ] normalisation
  - [x] get pixels
  - [x] coords to pixel
    - [x] improve indexing
- [ ] API
  - [ ] fix index error `lll one 155`
  - [x] config/setup/settings file with
    - nleds (or get automatically?)
  - [ ] Color module
    - [ ] calibrate colors (orange = yellow ?!)
    - [ ] conversions
    - [ ] documentation & docstrings
    - [x] color object
      - [x] use .get method in stead of bs now
  - [x] strip module
    - [ ] finish fancy indexing implementation
      - [ ] multiple color assignment?
      - [ ] multiple integers (tuple)
      - [ ] np.where compliant?
    - [x] integrate Color object
    - [ ] utils for plotting
    - [ ] automate moar
  - [ ] Pre-rendered animations
  - [ ] facilitate overlapping animations
- [ ] pkg
  - [ ] organise pkg
    - [x] connect git
  - [ ] documentation & docstrings
  - [ ] simple animations
    - [ ] Fireworks on sound
      - [ ] fireworks
        - [x] first version fireworks
      - [ ] sound
    - [ ] snow
  - [ ] examples
  - [ ] 3d model to animation conversion
  - [ ] virtual env for testing/animating
- expansion
  - [ ] controller on rpi
  - [ ] manymany ledled
  - [ ] mic
- [x] have fun
