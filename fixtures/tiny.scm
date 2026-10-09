;; tiny.scm — aura-loop fixture. `clamp` should keep x within [lo, hi].
;; tiny.tests.scm holds its Aura tests.
(define (clamp x lo hi)
  (max lo (min x)))
