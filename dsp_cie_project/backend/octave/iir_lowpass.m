% iir_lowpass.m
% Design and apply a low-pass IIR filter using second-order sections (SOS).
%
% Signature:
%   [y, sos, z, p, w, h_db] = iir_lowpass(x, fs, ftype, order, cutoff_hz,
%                                           ripple_db, stopband_db, zero_phase)
%
% Inputs:
%   x           - column vector of audio samples (float, [-1, 1])
%   fs          - sample rate in Hz
%   ftype       - 'butterworth', 'chebyshev', or 'elliptic'
%   order       - filter order (1 to 12)
%   cutoff_hz   - cutoff frequency in Hz (0 < cutoff < fs/2)
%   ripple_db   - passband ripple in dB (Chebyshev and Elliptic only)
%   stopband_db - stopband attenuation in dB (Elliptic only)
%   zero_phase  - logical; if true, filter forward then backward
%
% Outputs:
%   y     - filtered signal (column vector, same length as x)
%   sos   - Nx6 SOS matrix (same convention as scipy)
%   z     - zeros of the filter (complex column vector)
%   p     - poles of the filter (complex column vector)
%   w     - frequency axis in Hz (512-point, 0 to fs/2)
%   h_db  - magnitude response in dB at each point in w, floored at -120 dB

function [y, sos, z, p, w, h_db] = iir_lowpass(x, fs, ftype, order, cutoff_hz, ...
                                                 ripple_db, stopband_db, zero_phase)
  pkg load signal

  % Input validation
  if order < 1 || order > 12
    error('Order must be between 1 and 12.');
  end
  if cutoff_hz <= 0 || cutoff_hz >= fs / 2
    error('cutoff_hz must be between 0 and %g Hz (Nyquist).', fs / 2);
  end

  % Normalised cutoff (0 < Wn < 1, where 1 = Nyquist)
  Wn = cutoff_hz / (fs / 2);

  % Design using the [z, p, k] form, then convert to SOS.
  % The [b, a] form is numerically unstable at high orders; SOS keeps
  % each biquad well-conditioned regardless of order.
  switch lower(ftype)
    case 'butterworth'
      [z, p, k] = butter(order, Wn, 'low');
    case 'chebyshev'
      [z, p, k] = cheby1(order, ripple_db, Wn, 'low');
    case 'elliptic'
      [z, p, k] = ellip(order, ripple_db, stopband_db, Wn, 'low');
    otherwise
      error('Unknown filter type: %s', ftype);
  end

  % Convert zero-pole-gain to SOS and fold the overall gain g into the
  % first section so that cascade(sos) exactly equals the full filter.
  [sos, g] = zp2sos(z, p, k);
  sos(1, 1:3) = sos(1, 1:3) * g;

  % Apply the filter by cascading biquads one at a time.
  % Octave does not expose sosfilt, so we loop over sections manually.
  % For zero_phase=true we filter forward, flip, filter forward again, then
  % flip back. NOTE: this is NOT identical to SciPy's sosfiltfilt, which
  % pads the signal edges before filtering to reduce end transients. Small
  % differences at the start and end of the clip are therefore expected when
  % comparing the two engines at zero_phase=true.
  if zero_phase
    y = sos_apply(sos, x);
    y = flipud(y);
    y = sos_apply(sos, y);
    y = flipud(y);
  else
    y = sos_apply(sos, x);
  end

  % Frequency response: multiply the freqz response of each section over
  % 512 points from DC to Nyquist, then convert to dB.
  n_pts = 512;
  H = ones(n_pts, 1);
  for s = 1 : size(sos, 1)
    b_s = sos(s, 1:3);
    a_s = sos(s, 4:6);
    [h_s, w_rad] = freqz(b_s, a_s, n_pts);
    H = H .* h_s(:);
  end
  % w_rad goes from 0 to pi; convert to Hz
  w = (w_rad / pi) * (fs / 2);
  h_db = max(20 * log10(max(abs(H), 1e-6)), -120);

end % function iir_lowpass


% Helper: apply an SOS cascade to signal x using causal biquad filtering.
function y = sos_apply(sos, x)
  y = x(:);  % ensure column vector
  for s = 1 : size(sos, 1)
    b_s = sos(s, 1:3);
    a_s = sos(s, 4:6);
    y = filter(b_s, a_s, y);
  end
end
