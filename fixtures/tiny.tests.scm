;; Tests for fixtures/tiny.scm, run on Aura by SoftPlant against the post-mutation source.
;; (al-check name got want) is defined by aura_loop.plant.TEST_PRELUDE.
(al-check "clamp-high" (clamp 15 0 10) 10)
(al-check "clamp-low" (clamp -5 0 10) 0)
(al-check "clamp-mid" (clamp 5 0 10) 5)
