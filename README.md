
# litledlights

LitLedLights, they are Lit

## Resources

- adafruit
  - <https://learn.adafruit.com/neopixels-on-raspberry-pi/raspberry-pi-wiring>
  - <https://learn.adafruit.com/neopixels-on-raspberry-pi/python-usage>
  - [adafruit neopixel uberguide](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwisw86m_uz8AhVFM-wKHQVvAJgQFnoECA8QAQ&url=https%3A%2F%2Fcdn-learn.adafruit.com%2Fdownloads%2Fpdf%2Fadafruit-neopixel-uberguide.pdf&usg=AOvVaw1-UNr6xUSFV5fscJPYqsFR)

## To do

- level converter "chip"
- robust wiring

## Roadmap

- [x] setup rpi
- [x] link up stuff
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
  - [ ] organise pkg
    - [x] connect git
  - [ ] Color module
    - [ ] calibrate colors (orange = yellow ?!)
    - [ ] conversions
  - [x] strip module
    - [ ] finish fancy indexing implementation
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
  - [ ] simple animations
  - [ ] facilitate overlapping animations
  - [ ] examples
- [ ] virtual env for testing/animating
- expansion
  - [ ] controller on rpi
  - [ ] manymany ledled
- [x] have fun
