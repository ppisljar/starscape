// The demo consumes the library exactly as an external user would (DESIGN.md §8).
import { VERSION } from '../src/index';

const status = document.getElementById('status');
if (status) {
  status.textContent = `starscape v${VERSION} — toolchain live, renderer lands at stage 3`;
}
