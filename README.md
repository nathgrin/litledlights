
# litledlights

LitLedLights, they are Lit

## Resources

- adafruit
  - <https://learn.adafruit.com/neopixels-on-raspberry-pi/raspberry-pi-wiring>
  - <https://learn.adafruit.com/neopixels-on-raspberry-pi/python-usage>
  - [adafruit neopixel uberguide](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwisw86m_uz8AhVFM-wKHQVvAJgQFnoECA8QAQ&url=https%3A%2F%2Fcdn-learn.adafruit.com%2Fdownloads%2Fpdf%2Fadafruit-neopixel-uberguide.pdf&usg=AOvVaw1-UNr6xUSFV5fscJPYqsFR)
- Colors
  - [rgb/hsv colorpicker tool](https://math.hws.edu/graphicsbook/demos/c2/rgb-hsv.html)
  - [stackoverflow python functions](https://stackoverflow.com/questions/24852345/hsv-to-rgb-color-conversion)
## To do

- level converter "chip"
- robust wiring

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
    - [ ] rotate coords
    - [ ] fix wrong coords
    - [ ] fix missing coords
  - [ ] save and share coords (gift?)
- [ ] API
  - [ ] fix index error `lll one 155`
  - [ ] setup/settings file with
    - nleds (or get automatically?)
  - [ ] Color module
    - [ ] calibrate colors (orange = yellow ?!)
    - [ ] conversions
    - [ ] documentation
    - [x] color object
      - [ ] use .get method in stead of bs now

  - [x] strip module
    - [ ] finish fancy indexing implementation
      - [ ] multiple color assignment?
      - [ ] multiple integers (tuple)
      - [ ] np.where compliant?
    - [ ] integrate Color object
  - [ ] coord module
    - [ ] utils for plotting
    - [ ] automate moar
    - [ ] coordinate systems (spherical, cilindrical)
    - [ ] normalisation
  - [x] get pixels
  - [x] coords to pixel
    - [x] improve indexing
  - [ ] Pre-rendered animations
  - [ ] facilitate overlapping animations
- [ ] pkg
  - [ ] organise pkg
    - [x] connect git
  - [ ] simple animations
  - [ ] examples
  - [ ] 3d model to animation conversion
  - [ ] virtual env for testing/animating
- expansion
  - [ ] controller on rpi
  - [ ] manymany ledled
  - [ ] mic
- [x] have fun
