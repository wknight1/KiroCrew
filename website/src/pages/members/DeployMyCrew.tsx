/**
 * Your crew in the cloud: the Members page's answer to "is my crew deployed,
 * and what do I do next?"
 *
 * PERSONA FIRST. The crew's faces lead, then one plain sentence for the state
 * the crew is in, then the one thing a reader can do about it. The technical
 * identifiers an owner needs to debug a deploy (stack tag, provisioner id,
 * profile, region, instance id, raw status) are kept, but demoted to one
 * collapsed line at the bottom.
 *
 * SCOPE IS CREW-WIDE, NOT PER-MEMBER. A launch ships the whole local checkout
 * to one machine and names one CloudFormation stack (`tag`), so there is no
 * per-member deployment to show. That is why the trigger sits in the page
 * header beside "add a member" rather than inside a member's own drawer, and
 * why the faces here are the whole roster: the panel is about all of them at
 * once.
 *
 * ONE STATE, ONE ACTION. The newest launch decides which sentence is shown:
 *
 *   none        the crew runs only here            -> Deploy to the cloud
 *   deploying   step N of M                        -> nothing; wait
 *   signin      the launch needs the user's code   -> Finish the sign-in
 *   deployed    time since deploy, region          -> copy the address
 *   failed      the recorded error                 -> Deploy to the cloud
 *
 * Every navigation lands in Settings > Remote Instances, which owns creating,
 * repairing and tearing down a launch. This panel never writes.
 *
 * HONEST NUMBERS ONLY. Both numbers derive from the launch record the gateway
 * holds: the time since the launch was created, and its region. Sessions
 * running on the deployment and turns it served are deliberately absent: the
 * deployed side has no counter and no path to report back, and the local
 * activity log counts the LOCAL crew, so its numbers here would state a scope
 * the data does not have. An unknown value renders as an en dash, never as 0,
 * because "0" and "we could not read it" must never look the same.
 *
 * DELIBERATELY NOT the crew-summary dashboard's mechanism. That surface renders
 * content a CREW published, so it needs a sandboxed frame and a human-authored
 * template. This panel shows the operator their OWN cloud state, read by the
 * host from `GET /api/cloud/launch`, so it is ordinary trusted React: no
 * sandbox, no template engine, no publish path.
 */
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Check, Copy } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { api, type LaunchJob } from '../../api/client'
import CrewAvatar from '../../components/CrewAvatar'
import ErrorNotice from '../../components/ErrorNotice'
import { Btn } from '../../components/ui'
import {
  Dialog, DialogContent, DialogHeader, DialogBody, DialogFooter, DialogTitle,
} from '../../components/ui/dialog'
import { fmtDuration } from '../../i18n/format'
import { copyToClipboard } from '../../utils/clipboard'
import { BUILTIN_PROVISIONER_ID, FARGATE_PROVISIONER_ID, launchIsInFlight } from '../../utils/remoteCrew'

/** Where every action this panel offers lands: the Settings tab that owns the
 *  whole launch flow. */
const SETTINGS_PATH = '/settings/instances'

/** The unknown-value glyph, shared with the member stat cards: an en dash,
 *  never a zero, so "nothing" and "could not read" cannot render alike. */
const UNKNOWN = '\u2013'

/** How often the launch list is re-read while a launch is still moving. At
 *  rest nothing is polled: a finished or failed launch does not change. */
const IN_FLIGHT_POLL_MS = 4000

/** Faces drawn before the rest collapse into a "+N" tile. */
const MAX_FACES = 6

const LAUNCHES_QUERY_KEY = ['cloud', 'launches'] as const

/** The face of one crew member, as the roster carries it. */
export interface CrewFace {
  name: string
  avatar?: unknown
}

/**
 * Whether a launch has a machine a reader could reach right now.
 *
 * Only `done` qualifies. `running` is the launch still working, and a job that
 * `failed` or was `cancelled` can still carry an `instance_id` from the attempt
 * that got that far — offering it as reachable would send the reader at a
 * machine that is gone or half-built.
 */
export function deployIsReachable(job: Pick<LaunchJob, 'status' | 'instance_id'>): boolean {
  return job.status === 'done' && !!job.instance_id
}

/**
 * Whether the machine's id is something a reader can paste into Session
 * Manager. Only the built-in EC2 provisioner records an instance id there; the
 * Fargate lane records the task's ARN, which Session Manager does not accept,
 * so that launch is deployed and reachable but has no address to offer here.
 * `provider_id` is absent on records written before it existed, and those were
 * all EC2.
 */
export function deployHasShellTarget(
  job: Pick<LaunchJob, 'status' | 'instance_id' | 'provider_id'>,
): boolean {
  return deployIsReachable(job) && (job.provider_id ?? BUILTIN_PROVISIONER_ID) === BUILTIN_PROVISIONER_ID
}

/**
 * Whether the deployment is the Fargate lane's container task. This is what
 * the deployed state says in words where the target row would otherwise be.
 * Keyed on the Fargate id, not on "anything but EC2": a provisioner this panel
 * does not know is drawn verbatim in Details and gets no sentence, because
 * "runs as a container task" could be false of it. Nor is it the negation of
 * `deployHasShellTarget`: an EC2 launch that recorded no machine has no target
 * either, and gets no sentence for the same reason.
 */
export function deployRunsAsTask(job: Pick<LaunchJob, 'provider_id'>): boolean {
  return job.provider_id === FARGATE_PROVISIONER_ID
}

/**
 * The launch the panel is about: the most recently created one.
 *
 * Sorted here rather than trusting the list's order, so a gateway that returns
 * jobs in another order cannot make the panel describe a stale attempt as the
 * current one. Ties keep the list's own order.
 */
export function newestLaunch(jobs: readonly LaunchJob[]): LaunchJob | undefined {
  return [...jobs].sort((a, b) => b.created_at - a.created_at)[0]
}

/** What the panel says, derived from the newest launch alone. */
export type DeployView =
  | { kind: 'none' }
  | { kind: 'deploying'; job: LaunchJob }
  | { kind: 'signin'; job: LaunchJob }
  | { kind: 'deployed'; job: LaunchJob }
  | { kind: 'failed'; job: LaunchJob }

/**
 * Classify the launch list into the one state the panel shows.
 *
 * `awaiting_signin` is split out of the other in-flight statuses because it is
 * the one where waiting achieves nothing: the launch is holding for the user
 * to approve a code, and a panel that only said "deploying" would leave them
 * waiting on a step that is waiting on them.
 */
export function deployView(jobs: readonly LaunchJob[]): DeployView {
  const job = newestLaunch(jobs)
  if (!job) return { kind: 'none' }
  if (job.status === 'awaiting_signin') return { kind: 'signin', job }
  if (launchIsInFlight(job.status)) return { kind: 'deploying', job }
  if (job.status === 'done') return { kind: 'deployed', job }
  return { kind: 'failed', job }
}

/**
 * An older launch that still finished with a machine while the newest one did
 * not: the headline follows the newest launch, and this is what it would
 * otherwise bury. A retry that failed, or one still moving, does not tear the
 * previous stack down, so that machine keeps running (and billing) behind a
 * "did not finish" or "deploying" headline. Undefined when the newest launch is
 * itself the deployed one, or when no earlier launch is reachable.
 */
export function earlierReachableLaunch(jobs: readonly LaunchJob[]): LaunchJob | undefined {
  const newest = newestLaunch(jobs)
  if (!newest || deployIsReachable(newest)) return undefined
  return [...jobs]
    .filter((j) => j !== newest && deployIsReachable(j))
    .sort((a, b) => b.created_at - a.created_at)[0]
}

/**
 * "Step N of M" for a moving launch. The step list can be empty on a job an
 * older gateway persisted, so the total falls back to the four steps every
 * launch has (preflight, provision, sign-in, connect).
 */
export function deployProgress(job: Pick<LaunchJob, 'steps'>): { current: number; total: number } {
  const total = job.steps.length || 4
  const current = Math.min(total, job.steps.filter((s) => s.state === 'done').length + 1)
  return { current, total }
}

/**
 * The name of the step under way, so "Step 2 of 4" says what step 2 is. The
 * active step if the record marks one, else the first not yet done; the empty
 * string when the record carries no steps (an older gateway), in which case the
 * counter stands alone. The label is the gateway's own wording for the step.
 */
export function deployStepLabel(job: Pick<LaunchJob, 'steps'>): string {
  const active = job.steps.find((s) => s.state === 'active') ?? job.steps.find((s) => s.state !== 'done')
  return active?.label ?? ''
}

/**
 * Time since the launch was created, as a locale-aware "3d 4h" / "2h 10m" /
 * "7m". This is the age of the deploy record, which is what the gateway can
 * attest; it is labelled "since deploy" rather than "uptime" because nothing
 * here can see whether the machine ran the whole time.
 *
 * A missing or future timestamp renders as the unknown glyph rather than a
 * garbage age.
 */
export function deployAge(createdAtSec: number, nowMs = Date.now()): string {
  if (!Number.isFinite(createdAtSec) || createdAtSec <= 0) return UNKNOWN
  const ms = nowMs - createdAtSec * 1000
  if (ms < 0) return UNKNOWN
  const totalMin = Math.floor(ms / 60_000)
  const days = Math.floor(totalMin / 1440)
  const hours = Math.floor((totalMin % 1440) / 60)
  const minutes = totalMin % 60
  if (days >= 1) return fmtDuration([[days, 'day'], [hours, 'hour']], { dropZero: true })
  if (hours >= 1) return fmtDuration([[hours, 'hour'], [minutes, 'minute']], { dropZero: true })
  return fmtDuration([[minutes, 'minute']])
}

/** The crew, as faces. The whole roster, because the deployment is the whole
 *  roster; past `MAX_FACES` the rest collapse into a count. */
function CrewFaces({ members }: { members: readonly CrewFace[] }) {
  const shown = members.slice(0, MAX_FACES)
  const extra = members.length - shown.length
  if (shown.length === 0) return null
  return (
    <div className="flex items-center justify-center" data-testid="deploy-faces">
      {shown.map((m, i) => (
        <CrewAvatar
          key={m.name}
          seed={m.name}
          avatar={m.avatar}
          size={56}
          className={`rounded-lg ${i > 0 ? '-ml-3' : ''}`}
        />
      ))}
      {extra > 0 && (
        <span
          className="-ml-3 w-14 h-14 shrink-0 grid place-items-center rounded-lg border border-border bg-bg-elevated text-[13px] font-semibold text-muted"
          data-testid="deploy-faces-more"
        >
          +{extra}
        </span>
      )}
    </div>
  )
}

/** One stat card, copying the member stat vocabulary: a large number over an
 *  11px label, unknown drawn as the en dash. */
function Stat({ value, label, testid }: { value: string; label: string; testid: string }) {
  return (
    <div className="border border-border rounded-lg px-3 py-2">
      <div className="text-lg font-semibold leading-tight" data-testid={testid}>{value}</div>
      <div className="text-[11px] text-muted">{label}</div>
    </div>
  )
}

/**
 * The deployed crew's Session Manager target, with the one action the deployed
 * state offers. The value stays selectable text beside the button, so a reader
 * whose clipboard is denied can still read it, and a failed copy says so in
 * the shared error surface rather than leaving the button silent.
 */
function AddressRow({ value, onHandoff }: { value: string; onHandoff: () => void }) {
  const { t } = useTranslation()
  const [done, setDone] = useState(false)
  const [failed, setFailed] = useState(false)
  return (
    <div className="flex flex-col gap-1.5" data-testid="deploy-address">
      <div className="text-[11px] text-muted">{t('pages.membersPage.deploy_address')}</div>
      <div className="flex items-center gap-2">
        <code
          className="flex-1 min-w-0 break-all text-[12px] text-text bg-bg-hover rounded px-2 py-1"
          data-testid="deploy-address-value"
        >
          {value}
        </code>
        <Btn
          primary
          onClick={() => {
            // The guarded helper, not `navigator.clipboard` directly: on a
            // plain-HTTP remote dashboard the async API does not exist, and the
            // helper falls back to `execCommand` before reporting. It never
            // rejects, so the boolean is the whole outcome, and the tick is
            // gated on it: a tick over an unchanged clipboard is the worst
            // affordance there is.
            void copyToClipboard(value).then((ok) => {
              setDone(ok)
              setFailed(!ok)
            })
          }}
          data-testid="deploy-address-copy"
        >
          {done ? <Check size={14} /> : <Copy size={14} />}
          {done ? t('pages.membersPage.deploy_copied') : t('pages.membersPage.deploy_copy')}
        </Btn>
      </div>
      {/* askAgent ON: nothing here is unsaved, so the hand-off can destroy
          nothing, and the agent has a real remedy the message does not (read
          the id off the launch record and open the session itself). The dialog
          closes first, as for the panel's other two notices. */}
      {failed && (
        <ErrorNotice
          variant="inline"
          message={t('pages.membersPage.deploy_copy_failed')}
          askAgent
          onHandoff={onHandoff}
          testId="deploy-address-copy-error"
        />
      )}
      <div className="text-[11px] text-muted">{t('pages.membersPage.deploy_address_hint')}</div>
    </div>
  )
}

/**
 * The full-window panel.
 *
 * `open` is the host's state so the trigger and the panel do not both own it.
 * The query runs only while open (`enabled`), so a page visit that never opens
 * this panel makes no launch read at all; while a launch is moving it re-reads
 * every few seconds so the step count advances without a reopen.
 */
export default function DeployMyCrewDialog({
  open,
  onClose,
  members,
}: {
  open: boolean
  onClose: () => void
  members: readonly CrewFace[]
}) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const q = useQuery({
    queryKey: LAUNCHES_QUERY_KEY,
    queryFn: () => api.cloudLaunches(),
    enabled: open,
    refetchInterval: (query) =>
      query.state.data?.jobs.some((j) => launchIsInFlight(j.status)) ? IN_FLIGHT_POLL_MS : false,
  })
  const jobs = useMemo(() => q.data?.jobs ?? [], [q.data])
  const view = useMemo(() => deployView(jobs), [jobs])
  const earlier = useMemo(() => earlierReachableLaunch(jobs), [jobs])
  const ordered = useMemo(() => [...jobs].sort((a, b) => b.created_at - a.created_at), [jobs])

  // Closed first, then navigated: the destination is another page, so an open
  // dialog would otherwise be the first thing a reader sees when they come back.
  const goToSettings = () => {
    onClose()
    navigate(SETTINGS_PATH)
  }

  return (
    <Dialog open={open} onOpenChange={(v) => { if (!v) onClose() }}>
      <DialogContent maxWidth={520} className="max-h-[86vh]">
        <DialogHeader>
          <DialogTitle>{t('pages.membersPage.deploy_title')}</DialogTitle>
        </DialogHeader>
        <DialogBody className="flex flex-col gap-4">
          <CrewFaces members={members} />

          {q.isPending && (
            <div className="text-[12px] text-muted text-center" data-testid="deploy-loading">
              {t('pages.membersPage.deploy_loading')}
            </div>
          )}

          {/* A failed read is reported as unknown, never as "your crew runs
              only here": the second would tell a reader their crew is not
              deployed when the truth is that we could not find out. */}
          {q.isError && (
            <div className="flex flex-col items-center gap-2" data-testid="deploy-error">
              <ErrorNotice
                message={t('pages.membersPage.deploy_error')}
                askAgent
                onHandoff={onClose}
                className="w-full"
              />
              <Btn onClick={() => void q.refetch()} data-testid="deploy-retry">
                {t('pages.membersPage.deploy_retry')}
              </Btn>
            </div>
          )}

          {q.isSuccess && view.kind === 'none' && (
            <div className="flex flex-col items-center gap-3 text-center" data-testid="deploy-state-none">
              <p className="m-0 text-[14px] font-medium">{t('pages.membersPage.deploy_state_none')}</p>
              <Btn primary onClick={goToSettings} data-testid="deploy-action-deploy">
                {t('pages.membersPage.deploy_action_deploy')}
              </Btn>
              {/* The label names what the button does (open Settings) and where,
                  never the outcome a reader wants: a button that reads as the
                  act itself is not pressed by a reader who fears something is
                  created outside this computer right now. The line under it
                  says the one thing the label cannot: nothing is created until
                  the steps there are confirmed. */}
              <div className="text-[11.5px] text-muted" data-testid="deploy-action-hint">
                {t('pages.membersPage.deploy_action_hint')}
              </div>
            </div>
          )}

          {q.isSuccess && view.kind === 'deploying' && (
            <div className="flex flex-col items-center gap-1 text-center" data-testid="deploy-state-deploying">
              <p className="m-0 text-[14px] font-medium">{t('pages.membersPage.deploy_state_deploying')}</p>
              {/* "Step 2 of 4" says how far; the step's own name says what
                  step 2 is, in the gateway's wording. An older record with no
                  steps shows the counter alone. */}
              <div className="text-[12px] text-text" data-testid="deploy-progress">
                {t('pages.membersPage.deploy_step_progress', deployProgress(view.job))}
                {deployStepLabel(view.job) && (
                  <span className="text-muted" data-testid="deploy-step-label">
                    {' \u00b7 '}
                    {deployStepLabel(view.job)}
                  </span>
                )}
              </div>
              {/* No duration estimate: nothing in the repo measures a typical
                  launch, and a number nothing attests is not shown here (the
                  same rule the two stat cards follow). The step counter is
                  the progress. */}
              {/* The launch runs in the gateway, not in this dialog; a reader
                  who fears Close cancels it stays and babysits the window. */}
              <div className="text-[11.5px] text-muted" data-testid="deploy-close-hint">
                {t('pages.membersPage.deploy_close_hint')}
              </div>
            </div>
          )}

          {q.isSuccess && view.kind === 'signin' && (
            <div className="flex flex-col items-center gap-3 text-center" data-testid="deploy-state-signin">
              <p className="m-0 text-[14px] font-medium">{t('pages.membersPage.deploy_state_signin')}</p>
              <Btn primary onClick={goToSettings} data-testid="deploy-action-signin">
                {t('pages.membersPage.deploy_action_signin')}
              </Btn>
              {/* The same reassurance as the deploying state: the launch keeps
                  waiting in the gateway whether or not this window is open. */}
              <div className="text-[11.5px] text-muted" data-testid="deploy-close-hint">
                {t('pages.membersPage.deploy_close_hint')}
              </div>
            </div>
          )}

          {q.isSuccess && view.kind === 'deployed' && (
            <div className="flex flex-col gap-3" data-testid="deploy-state-deployed">
              <p className="m-0 text-[14px] font-medium text-center">
                {t('pages.membersPage.deploy_state_deployed')}
              </p>
              {/* Honest counters only: both come off the launch record. */}
              <div className="grid grid-cols-2 gap-2" data-testid="deploy-stats">
                <Stat
                  value={deployAge(view.job.created_at)}
                  label={t('pages.membersPage.deploy_stat_since')}
                  testid="deploy-stat-since"
                />
                <Stat
                  value={view.job.region || UNKNOWN}
                  label={t('pages.membersPage.deploy_stat_region')}
                  testid="deploy-stat-region"
                />
              </div>
              {/* The Fargate lane keeps a task ARN in `instance_id`, which is
                  not a Session Manager target; it stays in the Details line. */}
              {deployHasShellTarget(view.job) && view.job.instance_id && (
                <AddressRow value={view.job.instance_id} onHandoff={onClose} />
              )}
              {/* Said in words where the row would be, so the reader who saw
                  the row on an EC2 launch is not left guessing why this one
                  has none. Only for a provisioner that is not EC2: an EC2
                  launch with no machine recorded gets no sentence, because
                  "runs as a container task" would be false of it. */}
              {deployRunsAsTask(view.job) && (
                <div className="text-[11.5px] text-muted text-center" data-testid="deploy-no-target">
                  {t('pages.membersPage.deploy_no_target')}
                </div>
              )}
            </div>
          )}

          {q.isSuccess && view.kind === 'failed' && (
            <div className="flex flex-col items-center gap-3 text-center" data-testid="deploy-state-failed">
              <p className="m-0 text-[14px] font-medium">{t('pages.membersPage.deploy_state_failed')}</p>
              {/* The gateway's own sentence about what went wrong, verbatim,
                  through the shared error surface so the agent can take it. */}
              <ErrorNotice
                message={view.job.error}
                askAgent
                onHandoff={onClose}
                messageClassName="font-mono text-left"
                className="w-full"
                testId="deploy-launch-error"
              />
              <Btn primary onClick={goToSettings} data-testid="deploy-action-deploy">
                {t('pages.membersPage.deploy_action_deploy')}
              </Btn>
              <div className="text-[11.5px] text-muted" data-testid="deploy-action-hint">
                {t('pages.membersPage.deploy_action_hint')}
              </div>
            </div>
          )}

          {/* The headline follows the newest launch; this names what that would
              otherwise bury: an earlier launch that still runs (and bills)
              behind a failed retry or one still moving. */}
          {q.isSuccess && earlier && (
            <div className="text-[12px] text-text text-center" data-testid="deploy-earlier-live">
              {t('pages.membersPage.deploy_earlier_live', { region: earlier.region || UNKNOWN })}
            </div>
          )}

          {/* The identifiers an owner debugging a deploy needs, one line per
              launch, newest first, collapsed by default. The raw status token
              is deliberate here: it is the searchable name the gateway logs
              use, and the sentence above is the plain-language version. */}
          {ordered.length > 0 && (
            <details className="text-[11.5px] text-muted" data-testid="deploy-details">
              <summary className="cursor-pointer select-none">{t('pages.membersPage.details')}</summary>
              <ul className="list-none m-0 mt-1.5 p-0 space-y-1">
                {ordered.map((job) => (
                  <li key={job.id} className="font-mono text-[11px] break-words" data-testid="deploy-launch">
                    {[job.tag, job.provider_id, job.status, job.profile, job.region, job.instance_id]
                      .filter(Boolean)
                      .join(' \u00b7 ')}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </DialogBody>
        <DialogFooter>
          <Btn onClick={onClose} data-testid="deploy-close">
            {t('pages.membersPage.close')}
          </Btn>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
