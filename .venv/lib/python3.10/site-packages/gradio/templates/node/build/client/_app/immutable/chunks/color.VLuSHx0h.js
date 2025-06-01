import { w as ordered_colors } from "./2.DXfm8U3u.js";
const get_next_color = (index) => {
  return ordered_colors[index % ordered_colors.length];
};
export {
  get_next_color as g
};
