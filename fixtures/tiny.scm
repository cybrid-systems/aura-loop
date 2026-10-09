;; tiny.scm — Aura-loop MVP fixture.
;; Bug: `min` is called with one argument (arity error), so `clamp`
;; never applies the upper bound. The loop should repair it by rebinding
;; `clamp` with a precedent retrieved from memory.
(define (clamp x lo hi)
  (max lo (min x)))
